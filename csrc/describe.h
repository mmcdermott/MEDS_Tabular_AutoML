#ifndef MEDS_TAB_DESCRIBE_H_
#define MEDS_TAB_DESCRIBE_H_

#include <cstdint>
#include <deque>
#include <string>
#include <string_view>

#include <vector>

namespace meds_tab {

// One column of an Arrow record batch: its validity bitmap (null: no nulls), its data buffer and
// the array's offset in elements (applied to both).
struct ArrowColumn {
  const std::uint8_t* validity;
  const void* data;
  std::int64_t offset;
};

// The columns meds-tab-describe reads from one record batch of a MEDS shard.
struct ArrowRecords {
  std::int64_t length;
  ArrowColumn subjects;      // int64
  ArrowColumn times;         // Only the validity is read: a null time marks a static record.
  ArrowColumn code_indices;  // int32 indices into the dictionary
  ArrowColumn code_offsets;  // The dictionary's string offsets (int32, or int64 when large_offsets)
                             // and validity.
  const char* code_bytes;    // The dictionary's string data.
  std::int64_t dictionary_length;
  bool large_offsets;
  ArrowColumn values;  // float64 when values_64, else float32; NaN or null is missing.
  bool values_64;
};

// The four counts meds-tab-describe keeps per code, in this order.
enum DescribeCount : int {
  kCode = 0,
  kValue = 1,
  kStaticPresent = 2,
  kStaticMean = 3,
  kCodeCounts = 4
};

// meds-tab-describe's code frequencies over the records of training subjects: for every code,
// records with a subject and a code, split into timed records (kCode), timed records with a value
// (kValue), timeless records (kStaticPresent) and timeless records with a value (kStaticMean).
// Codes are interned once per distinct string of each batch's dictionary, never per record.
class CodeCounter {
 public:
  explicit CodeCounter(std::vector<std::int64_t> train_subjects);  // Sorted ascending.

  void Add(const ArrowRecords& records);
  // Whether any of subjects is a training subject (one search per run of equal subjects).
  bool HasTrainSubject(const std::int64_t* subjects, std::int64_t size) const;
  void Merge(const CodeCounter& other);
  // Adds the counts kept by dictionary entry to the codes' counts; size(), name() and counts() are
  // complete only after it (the C API flushes after every call that adds).
  void Flush();

  std::int64_t size() const { return static_cast<std::int64_t>(names_.size()); }
  const std::string& name(std::int64_t code) const {
    return names_[static_cast<std::size_t>(code)];
  }
  const std::vector<std::int64_t>& counts() const {
    return counts_;
  }  // size() x kCodeCounts, code-major.

 private:
  std::uint32_t Intern(std::string_view name);
  void Grow();
  template <typename Value>
  void CountRecords(const ArrowRecords& records);

  std::vector<std::int64_t> train_;
  std::deque<std::string> names_;  // Codes' names, by code.
  // Open addressing over names_: slots_ holds code + 1 (0: empty) and hashes_ each slot's name
  // hash. Every batch's dictionary is looked up whole, so lookups must be cheap: a word-at-a-time
  // hash and linear probing over flat arrays (std::hash<std::string_view> hashes byte by byte on
  // some platforms).
  std::vector<std::uint32_t> slots_;
  std::vector<std::uint64_t> hashes_;
  std::vector<std::int64_t> counts_;
  std::vector<std::uint32_t> remap_;  // The current batch's dictionary entries -> codes.
  std::vector<std::uint32_t> next_remap_;
  std::vector<std::int64_t> local_;  // The current batch's counts by dictionary entry.
  // The previous batch's dictionary (offsets rebased to 0, and bytes): consecutive row groups of a
  // shard usually repeat it, and then remap_ is reused after one comparison instead of a lookup per
  // entry.
  std::vector<std::int64_t> last_offsets_;
  std::string last_bytes_;
  bool last_valid_ = false;
};

}  // namespace meds_tab
#endif
