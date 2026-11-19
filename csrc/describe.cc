#include "describe.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace meds_tab {
namespace {

constexpr std::uint32_t kNoCode = std::numeric_limits<std::uint32_t>::max();

bool Valid(const ArrowColumn& column, std::int64_t i) {
  if (!column.validity) return true;
  const std::int64_t bit = column.offset + i;
  return (column.validity[bit >> 3] >> (bit & 7)) & 1;
}

template <typename T>
const T* Data(const ArrowColumn& column) {
  return static_cast<const T*>(column.data) + column.offset;
}

// A fast, well-mixed hash of a short string, eight bytes at a time. Equal names are still compared
// bytewise.
std::uint64_t Hash(std::string_view s) {
  std::uint64_t h = 0x9E3779B97F4A7C15ull ^ (s.size() * 0xFF51AFD7ED558CCDull);
  const char* p = s.data();
  std::size_t n = s.size();
  while (n >= 8) {
    std::uint64_t w;
    std::memcpy(&w, p, 8);
    h = (h ^ w) * 0xBF58476D1CE4E5B9ull;
    h ^= h >> 31;
    p += 8;
    n -= 8;
  }
  std::uint64_t w = 0;
  std::memcpy(&w, p, n);
  h = (h ^ w) * 0x94D049BB133111EBull;
  return h ^ (h >> 29);
}

}  // namespace

CodeCounter::CodeCounter(std::vector<std::int64_t> train_subjects)
    : train_(std::move(train_subjects)), slots_(1024, 0), hashes_(1024, 0) {
  if (!std::is_sorted(train_.begin(), train_.end())) {
    throw std::invalid_argument("Training subjects must be sorted.");
  }
}

void CodeCounter::Grow() {
  std::vector<std::uint32_t> slots(slots_.size() * 2, 0);
  std::vector<std::uint64_t> hashes(slots.size(), 0);
  const std::size_t mask = slots.size() - 1;
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    if (!slots_[i]) continue;
    std::size_t at = hashes_[i] & mask;
    while (slots[at]) at = (at + 1) & mask;
    slots[at] = slots_[i];
    hashes[at] = hashes_[i];
  }
  slots_.swap(slots);
  hashes_.swap(hashes);
}

std::uint32_t CodeCounter::Intern(std::string_view name) {
  const std::uint64_t hash = Hash(name);
  const std::size_t mask = slots_.size() - 1;
  std::size_t at = hash & mask;
  while (slots_[at]) {
    if (hashes_[at] == hash && names_[slots_[at] - 1] == name) return slots_[at] - 1;
    at = (at + 1) & mask;
  }
  if (names_.size() >= kNoCode - 1) throw std::length_error("Too many distinct codes.");
  const auto code = static_cast<std::uint32_t>(names_.size());
  names_.emplace_back(name);
  counts_.resize(counts_.size() + kCodeCounts, 0);
  slots_[at] = code + 1;
  hashes_[at] = hash;
  if (2 * names_.size() > slots_.size()) Grow();  // Load factor <= 1/2.
  return code;
}

void CodeCounter::Add(const ArrowRecords& r) {
  if (r.length < 0 || r.dictionary_length < 0) throw std::invalid_argument("Invalid batch length.");
  if (r.length > 0 && !(r.subjects.data && r.code_indices.data && r.values.data)) {
    throw std::invalid_argument("Missing record buffers.");
  }
  if (r.dictionary_length > 0 && !(r.code_offsets.data && r.code_bytes)) {
    throw std::invalid_argument("Missing dictionary buffers.");
  }
  // The dictionary's entries -> codes: one lookup per distinct string of the batch, unless the
  // dictionary repeats the previous batch's (compared whole: rebased offsets, then bytes).
  const auto offset = [&](std::int64_t d) -> std::int64_t {
    return r.large_offsets ? Data<std::int64_t>(r.code_offsets)[d]
                           : Data<std::int32_t>(r.code_offsets)[d];
  };
  const std::int64_t first = r.dictionary_length ? offset(0) : 0;
  const std::int64_t bytes = r.dictionary_length ? offset(r.dictionary_length) - first : 0;
  if (first < 0 || bytes < 0) throw std::invalid_argument("Invalid dictionary offsets.");
  bool same = last_valid_ && r.code_offsets.validity == nullptr &&
              static_cast<std::int64_t>(last_offsets_.size()) == r.dictionary_length + 1 &&
              static_cast<std::int64_t>(last_bytes_.size()) == bytes;
  for (std::int64_t d = 0; same && d <= r.dictionary_length; ++d)
    same = offset(d) - first == last_offsets_[d];
  same = same && (bytes == 0 || std::memcmp(last_bytes_.data(), r.code_bytes + first, bytes) == 0);
  if (!same) {
    Flush();  // The counts so far are by the previous dictionary's entries.
    // Row groups list codes in nearly the same order: an entry equal to the previous dictionary's
    // entry at the same position keeps its code without a lookup.
    next_remap_.assign(static_cast<std::size_t>(r.dictionary_length), kNoCode);
    const std::int64_t reusable =
        last_valid_ ? static_cast<std::int64_t>(last_offsets_.size()) - 1 : 0;
    for (std::int64_t d = 0; d < r.dictionary_length; ++d) {
      if (!Valid(r.code_offsets, d)) continue;
      const std::int64_t begin = offset(d), end = offset(d + 1);
      if (begin < first || end < begin || end - first > bytes) {
        throw std::invalid_argument("Invalid dictionary offsets.");
      }
      const std::string_view name(r.code_bytes + begin, static_cast<std::size_t>(end - begin));
      if (d < reusable) {
        const std::int64_t last_begin = last_offsets_[d], last_end = last_offsets_[d + 1];
        if (last_end - last_begin == end - begin &&
            std::memcmp(last_bytes_.data() + last_begin, name.data(), name.size()) == 0) {
          next_remap_[static_cast<std::size_t>(d)] = remap_[static_cast<std::size_t>(d)];
          continue;
        }
      }
      next_remap_[static_cast<std::size_t>(d)] = Intern(name);
    }
    remap_.swap(next_remap_);
    last_valid_ = r.code_offsets.validity == nullptr;
    if (last_valid_) {
      last_offsets_.resize(static_cast<std::size_t>(r.dictionary_length + 1));
      for (std::int64_t d = 0; d <= r.dictionary_length; ++d) last_offsets_[d] = offset(d) - first;
      last_bytes_.assign(r.code_bytes + first, static_cast<std::size_t>(bytes));
    }
    local_.assign(static_cast<std::size_t>(r.dictionary_length) * kCodeCounts, 0);
  }

  if (r.values_64) {
    CountRecords<double>(r);
  } else {
    CountRecords<float>(r);
  }
}

// The per-record pass, for one value width. Validity bitmaps are tested only for columns with
// nulls.
template <typename Value>
void CodeCounter::CountRecords(const ArrowRecords& r) {
  const auto* subjects = Data<std::int64_t>(r.subjects);
  const auto* indices = Data<std::int32_t>(r.code_indices);
  const auto* values = Data<Value>(r.values);
  // Counted per dictionary entry in a small dense array (cache-resident, unlike the counts of every
  // code), kept across batches with the same dictionary and added to the codes' counts by Flush().
  std::int64_t* local = local_.data();
  const bool subject_nulls = r.subjects.validity, code_nulls = r.code_indices.validity;
  const bool time_nulls = r.times.validity, value_nulls = r.values.validity;
  const std::uint64_t dictionary_length = static_cast<std::uint64_t>(r.dictionary_length);
  bool have_subject = false, is_train = false;
  std::int64_t subject = 0;
  for (std::int64_t i = 0; i < r.length; ++i) {
    if ((subject_nulls && !Valid(r.subjects, i)) || (code_nulls && !Valid(r.code_indices, i)))
      continue;
    if (!have_subject || subjects[i] != subject) {
      subject = subjects[i];
      have_subject = true;
      is_train = std::binary_search(train_.begin(), train_.end(), subject);
    }
    if (!is_train) continue;
    const auto index = static_cast<std::uint64_t>(static_cast<std::int64_t>(indices[i]));
    if (index >= dictionary_length) throw std::out_of_range("Code index out of the dictionary.");
    const bool timed = !time_nulls || Valid(r.times, i);
    const bool has_value = (!value_nulls || Valid(r.values, i)) && !std::isnan(values[i]);
    std::int64_t* cell = local + index * kCodeCounts + (timed ? kCode : kStaticPresent);
    cell[0] += 1;
    cell[1] += has_value;
  }
}

void CodeCounter::Flush() {
  const std::size_t entries = local_.size() / kCodeCounts;
  for (std::size_t d = 0; d < entries && d < remap_.size(); ++d) {
    const std::uint32_t code = remap_[d];
    std::int64_t* from = local_.data() + d * kCodeCounts;
    if (code != kNoCode) {  // A null dictionary entry is a null code: not counted.
      std::int64_t* to = counts_.data() + static_cast<std::size_t>(code) * kCodeCounts;
      for (int k = 0; k < kCodeCounts; ++k) to[k] += from[k];
    }
    for (int k = 0; k < kCodeCounts; ++k) from[k] = 0;
  }
}

bool CodeCounter::HasTrainSubject(const std::int64_t* subjects, std::int64_t size) const {
  if (size > 0 && !subjects) throw std::invalid_argument("Missing subjects.");
  for (std::int64_t i = 0; i < size; ++i) {
    if (i > 0 && subjects[i] == subjects[i - 1]) continue;
    if (std::binary_search(train_.begin(), train_.end(), subjects[i])) return true;
  }
  return false;
}

void CodeCounter::Merge(const CodeCounter& other) {
  for (std::int64_t code = 0; code < other.size(); ++code) {
    const std::uint32_t into = Intern(other.name(code));
    for (int k = 0; k < kCodeCounts; ++k) {
      counts_[static_cast<std::size_t>(into) * kCodeCounts + k] +=
          other.counts_[static_cast<std::size_t>(code) * kCodeCounts + k];
    }
  }
}

}  // namespace meds_tab
