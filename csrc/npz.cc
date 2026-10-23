#include "npz.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <vector>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "Blocks are stored little-endian.");

namespace meds_tab::npz {
namespace {

constexpr std::uint64_t kZip64Limit = (std::uint64_t{1} << 31) - 1;  // zipfile.ZIP64_LIMIT.
constexpr std::uint64_t kArrayHeaderBytes = 30 + 9 + 20;             // Header, name, ZIP64 extra.
constexpr std::uint16_t kDosDate = (1 << 5) | 1;           // 1980-01-01 00:00, as storage._member.
constexpr std::size_t kChunkBytes = std::size_t{1} << 20;  // Converted entries per write.
// Task rows are read as spans that also cover gaps up to kSpanGapBytes (one
// read instead of two per row); spans stop growing at kMaxSpanBytes.
constexpr std::int64_t kSpanGapBytes = std::int64_t{16} << 10;
constexpr std::int64_t kMaxSpanBytes = std::int64_t{8} << 20;

// Element types of the stored array, as NumPy dtypes.
enum class DType : std::uint8_t { kU1, kU2, kU4, kU8, kI1, kI2, kI4, kI8, kF4, kF8 };

struct TypeInfo {
  const char* descr;
  std::int64_t size;
  bool floating;
  bool is_signed;
};

constexpr TypeInfo kTypes[] = {{"|u1", 1, false, false}, {"<u2", 2, false, false},
                               {"<u4", 4, false, false}, {"<u8", 8, false, false},
                               {"|i1", 1, false, true},  {"<i2", 2, false, true},
                               {"<i4", 4, false, true},  {"<i8", 8, false, true},
                               {"<f4", 4, true, true},   {"<f8", 8, true, true}};

const TypeInfo& Info(DType type) { return kTypes[static_cast<int>(type)]; }

void Require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

// A malformed or mismatched source block.
void Check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

DType Integer(std::int64_t size, bool is_signed) {
  const int base = is_signed ? static_cast<int>(DType::kI1) : 0;
  const int step = size == 1 ? 0 : size == 2 ? 1 : size == 4 ? 2 : 3;
  return static_cast<DType>(base + step);
}

// numpy.result_type of two types.
DType Promote(DType a, DType b) {
  if (a == b) return a;
  const TypeInfo &x = Info(a), &y = Info(b);
  if (x.floating || y.floating) {
    if (x.floating && y.floating) return DType::kF8;
    const TypeInfo& f = x.floating ? x : y;
    const TypeInfo& i = x.floating ? y : x;
    return f.size == 8 || i.size > 2 ? DType::kF8 : DType::kF4;
  }
  if (x.is_signed == y.is_signed) return x.size >= y.size ? a : b;
  const TypeInfo& s = x.is_signed ? x : y;
  const TypeInfo& u = x.is_signed ? y : x;
  if (u.size < s.size) return x.is_signed ? a : b;
  return u.size == 8 ? DType::kF8 : Integer(2 * u.size, true);
}

// numpy.min_scalar_type of an integer.
DType MinScalar(std::int64_t value) {
  if (value >= 0) {
    return value <= 0xFF         ? DType::kU1
           : value <= 0xFFFF     ? DType::kU2
           : value <= 0xFFFFFFFF ? DType::kU4
                                 : DType::kU8;
  }
  return value >= -0x80       ? DType::kI1
         : value >= -0x8000   ? DType::kI2
         : value >= INT32_MIN ? DType::kI4
                              : DType::kI8;
}

// Calls function(T{}) with the C++ type of a stored type.
template <typename Function>
void Dispatch(DType type, Function function) {
  switch (type) {
    case DType::kU1:
      return function(std::uint8_t{});
    case DType::kU2:
      return function(std::uint16_t{});
    case DType::kU4:
      return function(std::uint32_t{});
    case DType::kU8:
      return function(std::uint64_t{});
    case DType::kI1:
      return function(std::int8_t{});
    case DType::kI2:
      return function(std::int16_t{});
    case DType::kI4:
      return function(std::int32_t{});
    case DType::kI8:
      return function(std::int64_t{});
    case DType::kF4:
      return function(float{});
    case DType::kF8:
      return function(double{});
  }
}

void Put(std::string& out, std::uint64_t value, int bytes) {
  for (int i = 0; i < bytes; ++i) out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

std::uint64_t Get(const unsigned char* data, int bytes) {
  std::uint64_t value = 0;
  for (int i = bytes - 1; i >= 0; --i) value = (value << 8) | data[i];
  return value;
}

template <typename T>
T Load(const unsigned char* data) {
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}

std::uint32_t Crc(std::uint32_t crc, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const Bytef*>(data);
  while (size) {
    const auto piece = static_cast<uInt>(std::min<std::size_t>(size, std::size_t{1} << 30));
    crc = static_cast<std::uint32_t>(crc32(crc, bytes, piece));
    bytes += piece;
    size -= piece;
  }
  return crc;
}

// numpy.lib.format._write_array_header, version 1.0: the sorted header dict,
// room to grow the first axis (here one digit) to 21 digits, then spaces and a
// newline up to a multiple of 64 bytes.
std::string NpyHeader(const char* descr, const std::string& shape) {
  std::string dict =
      std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
  dict.append(21 - 1, ' ');
  const std::size_t length = dict.size() + 1;
  const std::size_t padding = 64 - (10 + length) % 64;
  std::string out("\x93NUMPY\x01\x00", 8);
  Put(out, length + padding, 2);
  out += dict;
  out.append(padding, ' ');
  out += '\n';
  return out;
}

// zipfile.ZipInfo.FileHeader for a stored member dated 1980-01-01.
std::string LocalHeader(const char* name, std::uint32_t crc, std::uint64_t size, bool zip64) {
  std::string out("PK\x03\x04", 4);
  Put(out, zip64 ? 45 : 20, 1);  // Version needed to extract.
  Put(out, 0, 1);
  Put(out, 0, 2);  // Flags.
  Put(out, 0, 2);  // Stored.
  Put(out, 0, 2);  // Time.
  Put(out, kDosDate, 2);
  Put(out, crc, 4);
  const std::uint64_t field = zip64 ? 0xFFFFFFFF : size;
  Put(out, field, 4);  // Compressed size.
  Put(out, field, 4);  // Size.
  Put(out, std::strlen(name), 2);
  Put(out, zip64 ? 20 : 0, 2);
  out += name;
  if (zip64) {
    Put(out, 1, 2);
    Put(out, 16, 2);
    Put(out, size, 8);
    Put(out, size, 8);
  }
  return out;
}

// One central directory record (zipfile.ZipFile._write_end_record).
void CentralRecord(std::string& out, const char* name, std::uint32_t crc, std::uint64_t size,
                   std::uint64_t offset, int version) {
  std::vector<std::uint64_t> extra;
  if (size > kZip64Limit) extra.insert(extra.end(), {size, size});
  if (offset > kZip64Limit) extra.push_back(offset);
  if (!extra.empty()) version = 45;
  out.append("PK\x01\x02", 4);
  Put(out, version, 1);  // Created by this version...
  Put(out, 3, 1);        // ...on Unix.
  Put(out, version, 1);  // Version needed to extract.
  Put(out, 0, 1);
  Put(out, 0, 2);  // Flags.
  Put(out, 0, 2);  // Stored.
  Put(out, 0, 2);  // Time.
  Put(out, kDosDate, 2);
  Put(out, crc, 4);
  const std::uint64_t field = size > kZip64Limit ? 0xFFFFFFFF : size;
  Put(out, field, 4);
  Put(out, field, 4);
  Put(out, std::strlen(name), 2);
  Put(out, extra.empty() ? 0 : 4 + 8 * extra.size(), 2);
  Put(out, 0, 2);            // Comment.
  Put(out, 0, 2);            // Disk.
  Put(out, 0, 2);            // Internal attributes.
  Put(out, 0600u << 16, 4);  // External attributes: ?rw-------.
  Put(out, offset > kZip64Limit ? 0xFFFFFFFF : offset, 4);
  out += name;
  if (!extra.empty()) {
    Put(out, 1, 2);
    Put(out, 8 * extra.size(), 2);
    for (const auto value : extra) Put(out, value, 8);
  }
}

class OutputFile {
 public:
  explicit OutputFile(const std::string& path) : path_(path) {
    // Create missing parent directories, as write_csr_block does.
    for (auto slash = path.find('/', 1); slash != std::string::npos;
         slash = path.find('/', slash + 1)) {
      if (::mkdir(path.substr(0, slash).c_str(), 0777) != 0 && errno != EEXIST) Fail("mkdir");
    }
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
    if (fd_ < 0) Fail("open");
  }
  OutputFile(const OutputFile&) = delete;
  OutputFile& operator=(const OutputFile&) = delete;
  ~OutputFile() {
    if (fd_ >= 0) ::close(fd_);
  }

  void Write(const void* data, std::size_t size) {
    const char* bytes = static_cast<const char*>(data);
    while (size) {
      const auto written = ::write(fd_, bytes, std::min(size, kChunkBytes));
      if (written < 0) {
        if (errno == EINTR) continue;
        Fail("write");
      }
      bytes += written;
      size -= static_cast<std::size_t>(written);
    }
  }
  void Write(const std::string& data) { Write(data.data(), data.size()); }

  void WriteAt(const std::string& data, off_t offset) {
    std::size_t done = 0;
    while (done < data.size()) {
      const auto written =
          ::pwrite(fd_, data.data() + done, data.size() - done, offset + static_cast<off_t>(done));
      if (written < 0) {
        if (errno == EINTR) continue;
        Fail("write");
      }
      done += static_cast<std::size_t>(written);
    }
  }

  void Close() {
    const int fd = fd_;
    fd_ = -1;
    if (::close(fd) != 0) Fail("close");
  }

 private:
  [[noreturn]] void Fail(const char* operation) {
    throw std::system_error(errno, std::generic_category(), path_ + ": " + operation);
  }

  std::string path_;
  int fd_ = -1;
};

class InputFile {
 public:
  explicit InputFile(const std::string& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), path + ": open");
  }
  InputFile(const InputFile&) = delete;
  InputFile& operator=(const InputFile&) = delete;
  ~InputFile() { ::close(fd_); }

  std::int64_t Size() const {
    struct stat status;
    if (::fstat(fd_, &status) != 0) {
      throw std::system_error(errno, std::generic_category(), path_ + ": stat");
    }
    return static_cast<std::int64_t>(status.st_size);
  }

  // Reads up to size bytes at offset; returns the count (short only at the end).
  std::size_t ReadSome(void* data, std::size_t size, std::int64_t offset) const {
    char* bytes = static_cast<char*>(data);
    std::size_t done = 0;
    while (done < size) {
      const auto read = ::pread(fd_, bytes + done, std::min(size - done, kChunkBytes << 6),
                                static_cast<off_t>(offset) + static_cast<off_t>(done));
      if (read < 0) {
        if (errno == EINTR) continue;
        throw std::system_error(errno, std::generic_category(), path_ + ": read");
      }
      if (read == 0) break;
      done += static_cast<std::size_t>(read);
    }
    return done;
  }

  // Starts reading a range the caller will read soon (advisory; errors ignored),
  // so the reads of many ranges are in flight together.
  void Advise(std::int64_t offset, std::int64_t size) const {
#if defined(__APPLE__)
    radvisory advice{static_cast<off_t>(offset), static_cast<int>(size)};
    ::fcntl(fd_, F_RDADVISE, &advice);
#elif defined(POSIX_FADV_WILLNEED)
    ::posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(size), POSIX_FADV_WILLNEED);
#endif
  }

  void Read(void* data, std::size_t size, std::int64_t offset) const {
    Check(ReadSome(data, size, offset) == size, path_ + ": block is truncated.");
  }

  const std::string& path() const { return path_; }

 private:
  std::string path_;
  int fd_ = -1;
};

// A parsed .npy header (versions 1.0 and 2.0).
struct Npy {
  DType dtype = DType::kU1;
  std::vector<std::int64_t> shape;
  std::int64_t header = 0;  // Bytes before the elements.
};

// Bytes of the .npy preamble and header starting at data (size available), or
// 0 when more than size bytes are needed to know.
std::int64_t NpyHeaderBytes(const unsigned char* data, std::size_t size) {
  if (size < 12) return 0;
  Check(std::memcmp(data, "\x93NUMPY", 6) == 0, "Not a .npy member.");
  if (data[6] == 1 && data[7] == 0) return 10 + static_cast<std::int64_t>(Get(data + 8, 2));
  Check(data[6] == 2 && data[7] == 0, "Unsupported .npy version.");
  return 12 + static_cast<std::int64_t>(Get(data + 8, 4));
}

Npy ParseNpy(const unsigned char* data, std::size_t size) {
  Npy npy;
  npy.header = NpyHeaderBytes(data, size);
  Check(npy.header > 0 && static_cast<std::size_t>(npy.header) <= size, "Truncated .npy header.");
  const std::size_t start = data[6] == 1 ? 10 : 12;
  const std::string text(reinterpret_cast<const char*>(data) + start,
                         static_cast<std::size_t>(npy.header) - start);
  const auto value = [&](const char* key) {
    const auto at = text.find(key);
    Check(at != std::string::npos, "Incomplete .npy header.");
    auto position = text.find_first_not_of(' ', at + std::strlen(key));
    Check(position != std::string::npos, "Incomplete .npy header.");
    return position;
  };
  auto position = value("'descr':");
  Check(text[position] == '\'', "Unsupported .npy descr.");
  const auto end = text.find('\'', position + 1);
  Check(end != std::string::npos, "Unsupported .npy descr.");
  const std::string descr = text.substr(position + 1, end - position - 1);
  bool found = false;
  for (int t = 0; t < 10; ++t) {
    if (descr == kTypes[t].descr) {
      npy.dtype = static_cast<DType>(t);
      found = true;
    }
  }
  Check(found, "Unsupported stored type " + descr + ".");
  position = value("'fortran_order':");
  Check(text.compare(position, 5, "False") == 0, "Fortran-ordered arrays are unsupported.");
  position = value("'shape':");
  Check(text[position] == '(', "Unsupported .npy shape.");
  ++position;
  while (true) {
    position = text.find_first_not_of(' ', position);
    Check(position != std::string::npos, "Unsupported .npy shape.");
    if (text[position] == ')') break;
    // At most 18 digits, so the dimension cannot overflow int64.
    const auto is_digit = [&] {
      return position < text.size() && text[position] >= '0' && text[position] <= '9';
    };
    std::size_t digits = 0;
    std::int64_t dimension = 0;
    for (; is_digit() && digits < 18; ++digits)
      dimension = dimension * 10 + (text[position++] - '0');
    Check(digits > 0 && !is_digit(), "Unsupported .npy shape.");
    npy.shape.push_back(dimension);
    position = text.find_first_not_of(' ', position);
    Check(position != std::string::npos, "Unsupported .npy shape.");
    if (text[position] == ',') ++position;
  }
  return npy;
}

struct Layout {
  std::int64_t offset = 0;  // Of the first element.
  DType dtype = DType::kU1;
  std::int64_t nnz = 0;
};

// Where the elements of 'array.npy' start. It must be the file's first member,
// stored uncompressed, as WriteBlock writes it.
Layout StoredLayout(const InputFile& file) {
  std::vector<unsigned char> head(4096);
  head.resize(file.ReadSome(head.data(), head.size(), 0));
  Check(head.size() >= 30 && std::memcmp(head.data(), "PK\x03\x04", 4) == 0 &&
            Get(head.data() + 8, 2) == 0,
        file.path() + ": not a stored block.");
  const auto name = Get(head.data() + 26, 2);
  const auto start = 30 + name + Get(head.data() + 28, 2);
  Check(name == 9 && head.size() >= 39 && std::memcmp(head.data() + 30, "array.npy", 9) == 0,
        file.path() + ": not a stored block.");
  Check(head.size() >= start, file.path() + ": block is truncated.");
  auto header = NpyHeaderBytes(head.data() + start, head.size() - start);
  if (header == 0 || start + static_cast<std::uint64_t>(header) > head.size()) {
    head.resize(start + 12);
    file.Read(head.data(), head.size(), 0);
    header = NpyHeaderBytes(head.data() + start, 12);
    head.resize(start + static_cast<std::uint64_t>(header));
    file.Read(head.data(), head.size(), 0);
  }
  const Npy npy = ParseNpy(head.data() + start, head.size() - start);
  Check(npy.shape.size() == 2 && npy.shape[0] == 3,
        file.path() + ": 'array' must be a C-ordered (3, nnz) array.");
  // The array must lie inside the file, which also bounds every byte offset computed from nnz.
  const std::int64_t data = static_cast<std::int64_t>(start) + npy.header, nnz = npy.shape[1];
  Check(nnz <= (file.Size() - data) / (3 * Info(npy.dtype).size),
        file.path() + ": block is truncated.");
  return {data, npy.dtype, nnz};
}

// The local header of 'array.npy', written first with a zero CRC and rewritten
// once the payload's CRC is known.
std::string ArrayHeader(std::uint32_t crc, std::uint64_t payload) {
  return LocalHeader("array.npy", crc, payload, true);
}

// Every byte after the 'array.npy' payload: 'shape.npy', the central directory
// and the end records.
std::string Trailer(std::uint32_t crc, std::uint64_t payload, std::int64_t rows,
                    std::int64_t columns) {
  std::string shape = NpyHeader("<i8", "(2,)");
  Put(shape, static_cast<std::uint64_t>(rows), 8);
  Put(shape, static_cast<std::uint64_t>(columns), 8);
  const auto shape_crc = Crc(0, shape.data(), shape.size());
  const std::uint64_t shape_offset = kArrayHeaderBytes + payload;
  std::string out = LocalHeader("shape.npy", shape_crc, shape.size(), false) + shape;
  const std::uint64_t start = shape_offset + out.size();
  const auto records = out.size();
  CentralRecord(out, "array.npy", crc, payload, 0, 45);
  CentralRecord(out, "shape.npy", shape_crc, shape.size(), shape_offset, 20);
  const std::uint64_t size = out.size() - records;
  std::uint64_t offset = start;
  if (start > kZip64Limit) {
    out.append("PK\x06\x06", 4);  // ZIP64 end of central directory.
    Put(out, 44, 8);
    Put(out, 45, 2);
    Put(out, 45, 2);
    Put(out, 0, 4);
    Put(out, 0, 4);
    Put(out, 2, 8);
    Put(out, 2, 8);
    Put(out, size, 8);
    Put(out, start, 8);
    out.append("PK\x06\x07", 4);  // Its locator.
    Put(out, 0, 4);
    Put(out, start + size, 8);
    Put(out, 1, 4);
    offset = std::min<std::uint64_t>(offset, 0xFFFFFFFF);
  }
  out.append("PK\x05\x06", 4);
  Put(out, 0, 2);
  Put(out, 0, 2);
  Put(out, 2, 2);
  Put(out, 2, 2);
  Put(out, size, 4);
  Put(out, offset, 4);
  Put(out, 0, 2);
  return out;
}

// Writes count elements of type T, produced chunk by chunk by
// fill(buffer, begin, n), and folds them into the running CRC.
template <typename T, typename Fill>
void Stream(OutputFile& file, std::uint32_t& crc, std::int64_t count, Fill fill) {
  constexpr std::int64_t step = static_cast<std::int64_t>(kChunkBytes / sizeof(T));
  if (count <= 0) return;
  std::unique_ptr<T[]> buffer(new T[static_cast<std::size_t>(std::min(step, count))]);
  for (std::int64_t begin = 0; begin < count; begin += step) {
    const auto n = std::min(step, count - begin);
    fill(buffer.get(), begin, n);
    const auto bytes = static_cast<std::size_t>(n) * sizeof(T);
    crc = Crc(crc, buffer.get(), bytes);
    file.Write(buffer.get(), bytes);
  }
}

std::int64_t ToInt64(double value) {
  Require(std::isfinite(value) && value >= -0x1p63 && value < 0x1p63,
          "Integer feature values must be finite and within int64.");
  return static_cast<std::int64_t>(value);
}
std::int64_t ToInt64(std::int64_t value) { return value; }

// write_csr_block's float32 test: the value survives conversion to float32.
bool ExactFloat(double value) {
  return std::fabs(value) <= FLT_MAX && static_cast<double>(static_cast<float>(value)) == value;
}
template <typename Value>
bool ExactFloat(Value value) {
  return static_cast<double>(static_cast<float>(value)) == static_cast<double>(value);
}

template <typename Index, typename Value>
std::int64_t WriteTyped(const std::string& path, const Block& block, bool integer) {
  const auto* outer = static_cast<const Index*>(block.outer);
  const auto* inner = static_cast<const Index*>(block.inner);
  const auto* values = static_cast<const Value*>(block.values);
  const std::int64_t rows = block.rows, columns = block.columns, nnz = block.nnz;
  Require(rows >= 0 && columns >= 0 && nnz >= 0, "Invalid block dimensions.");
  Require(outer && (nnz == 0 || (inner && values)), "Missing block arrays.");
  Require(outer[0] == 0 && static_cast<std::int64_t>(outer[rows]) == nnz,
          "Row offsets must run from 0 to the entry count.");
  for (std::int64_t r = 0; r < rows; ++r) {
    Require(outer[r] <= outer[r + 1], "Row offsets must be nondecreasing.");
  }

  // The stored type, as storage.value_dtype and homogeneous_dtype choose it: the
  // smallest exact type for the values, promoted with the smallest types of the
  // row and column IDs.
  DType value_type;
  if (integer) {
    value_type = DType::kU1;
    if (nnz) {
      std::int64_t low = ToInt64(values[0]), high = low;
      for (std::int64_t i = 1; i < nnz; ++i) {
        const auto v = ToInt64(values[i]);
        low = std::min(low, v);
        high = std::max(high, v);
      }
      value_type = Promote(MinScalar(low), MinScalar(high));
    }
  } else {
    bool narrow = true;
    for (std::int64_t i = 0; i < nnz; ++i) {
      if constexpr (std::is_floating_point_v<Value>) {
        Require(std::isfinite(values[i]), "Stored feature values must be finite.");
      }
      narrow = narrow && ExactFloat(values[i]);
    }
    value_type = narrow ? DType::kF4 : DType::kF8;
  }
  DType common = Promote(Promote(value_type, MinScalar(std::max<std::int64_t>(rows - 1, 0))),
                         MinScalar(std::max<std::int64_t>(columns - 1, 0)));
  if (Info(common).floating) {
    const std::int64_t max_index = std::max({rows - 1, columns - 1, std::int64_t{0}});
    if (max_index >= (std::int64_t{1} << 24)) common = DType::kF8;
    Require(max_index < (std::int64_t{1} << 53),
            "Row and column IDs exceed exact float64 integers.");
  }

  const std::string header = NpyHeader(Info(common).descr, "(3, " + std::to_string(nnz) + ")");
  const std::uint64_t payload = header.size() + 3 * static_cast<std::uint64_t>(nnz) *
                                                    static_cast<std::uint64_t>(Info(common).size);
  OutputFile file(path);
  file.Write(ArrayHeader(0, payload));  // Rewritten with the CRC at the end.
  std::uint32_t crc = Crc(0, header.data(), header.size());
  file.Write(header);
  Dispatch(common, [&](auto stored) {
    using T = decltype(stored);
    Stream<T>(file, crc, nnz, [&](T* buffer, std::int64_t begin, std::int64_t n) {
      for (std::int64_t i = 0; i < n; ++i) buffer[i] = static_cast<T>(values[begin + i]);
    });
    std::int64_t row = 0;
    Stream<T>(file, crc, nnz, [&](T* buffer, std::int64_t begin, std::int64_t n) {
      for (std::int64_t i = 0; i < n;) {
        while (static_cast<std::int64_t>(outer[row + 1]) <= begin + i) ++row;
        const auto end = std::min(n, static_cast<std::int64_t>(outer[row + 1]) - begin);
        std::fill(buffer + i, buffer + end, static_cast<T>(row));
        i = end;
      }
    });
    Stream<T>(file, crc, nnz, [&](T* buffer, std::int64_t begin, std::int64_t n) {
      for (std::int64_t i = 0; i < n; ++i) {
        const auto column = static_cast<std::int64_t>(inner[begin + i]);
        Require(column >= 0 && column < columns, "Entry columns must lie within the block.");
        buffer[i] = static_cast<T>(column);
      }
    });
  });
  const std::string trailer = Trailer(crc, payload, rows, columns);
  file.Write(trailer);
  file.WriteAt(ArrayHeader(crc, payload), 0);
  file.Close();
  return static_cast<std::int64_t>(kArrayHeaderBytes + payload + trailer.size());
}

// An entry range of the source block and where it goes in the output.
struct Range {
  std::int64_t start;
  std::int64_t length;
  std::int64_t output;
};

// Reads the given entry ranges of a stored block of element type S into values
// and columns, checking every column against the block's width. Ranges sorted by
// start are read as spans that also cover small gaps, so scattered rows cost a
// few large reads.
template <typename S, typename Value>
void Gather(const InputFile& file, const Layout& layout, const std::vector<Range>& ranges,
            std::int64_t columns, Value* values, std::int64_t* column_ids) {
  const std::int64_t item = sizeof(S);
  const std::int64_t gap = kSpanGapBytes / item, max_span = kMaxSpanBytes / item;
  struct Span {
    std::int64_t begin, end;
    std::size_t first, last;  // Ranges [first, last).
  };
  std::vector<Span> spans;
  for (std::size_t first = 0; first < ranges.size();) {
    const std::int64_t begin = ranges[first].start;
    std::int64_t end = begin + ranges[first].length;
    std::size_t last = first + 1;
    for (; last < ranges.size(); ++last) {
      const auto next = ranges[last].start + ranges[last].length;
      if (ranges[last].start > end + gap || std::max(end, next) - begin > max_span) break;
      end = std::max(end, next);
    }
    spans.push_back({begin, end, first, last});
    first = last;
  }
  // Blocks just written may have left the page cache (macOS keeps roughly the
  // last gigabyte written); request every span before reading any.
  for (const auto& span : spans) {
    file.Advise(layout.offset + span.begin * item, (span.end - span.begin) * item);
    file.Advise(layout.offset + (2 * layout.nnz + span.begin) * item,
                (span.end - span.begin) * item);
  }
  std::vector<S> data, ids;
  for (const auto& span : spans) {
    const std::int64_t begin = span.begin, end = span.end;
    const std::size_t first = span.first, last = span.last;
    const auto count = static_cast<std::size_t>(end - begin);
    data.resize(count);
    ids.resize(count);
    file.Read(data.data(), count * sizeof(S), layout.offset + begin * item);
    file.Read(ids.data(), count * sizeof(S), layout.offset + (2 * layout.nnz + begin) * item);
    for (std::size_t k = first; k < last; ++k) {
      const auto at = static_cast<std::size_t>(ranges[k].start - begin);
      for (std::int64_t j = 0; j < ranges[k].length; ++j) {
        const S value = data[at + j], id = ids[at + j];
        const auto out = ranges[k].output + j;
        if constexpr (std::is_floating_point_v<S>) {
          Check(std::isfinite(id) && id >= 0 && id < static_cast<double>(columns),
                file.path() + ": entry indices are outside the stored shape.");
          if constexpr (std::is_same_v<Value, std::int64_t>) {
            Check(std::isfinite(value) && std::fabs(value) < 0x1p63,
                  file.path() + ": integer values out of range.");
          }
        } else {
          Check(static_cast<std::int64_t>(id) >= 0 && static_cast<std::int64_t>(id) < columns,
                file.path() + ": entry indices are outside the stored shape.");
        }
        column_ids[out] = static_cast<std::int64_t>(id);
        values[out] = static_cast<Value>(value);
      }
    }
  }
}

}  // namespace

std::int64_t WriteBlock(const std::string& path, const Block& block, bool integer) {
  const auto typed = [&](auto index) {
    using Index = decltype(index);
    return block.values_integer ? WriteTyped<Index, std::int64_t>(path, block, integer)
                                : WriteTyped<Index, double>(path, block, integer);
  };
  return block.index_64 ? typed(std::int64_t{}) : typed(std::int32_t{});
}

std::int64_t ExtractRows(const std::string& source, const std::int64_t* outer,
                         std::int64_t source_rows, std::int64_t columns, const std::int64_t* rows,
                         std::int64_t row_count, bool integer, const std::string& target,
                         std::int64_t* written_outer) {
  Require(outer && row_count >= 0 && (row_count == 0 || rows) && columns >= 0,
          "Invalid extraction arguments.");
  const InputFile file(source);
  const Layout layout = StoredLayout(file);
  Check(source_rows >= 0 && outer[source_rows] == layout.nnz,
        source + ": recorded row offsets do not match the block.");
  std::vector<std::int64_t> selected(static_cast<std::size_t>(row_count) + 1, 0);
  std::vector<Range> ranges;
  for (std::int64_t i = 0; i < row_count; ++i) {
    const auto r = rows[i];
    Check(r < source_rows, source + ": requested rows are outside the stored shape.");
    std::int64_t length = 0;
    if (r >= 0) {
      Check(outer[r] >= 0 && outer[r] <= outer[r + 1] && outer[r + 1] <= layout.nnz,
            source + ": recorded row offsets do not match the block.");
      length = outer[r + 1] - outer[r];
      if (length) ranges.push_back({outer[r], length, selected[i]});
    }
    selected[i + 1] = selected[i] + length;
  }
  std::sort(ranges.begin(), ranges.end(),
            [](const Range& a, const Range& b) { return a.start < b.start; });
  const auto total = static_cast<std::size_t>(selected.back());
  std::vector<std::int64_t> column_ids(total);
  std::vector<double> doubles(integer ? 0 : total);
  std::vector<std::int64_t> integers(integer ? total : 0);
  Dispatch(layout.dtype, [&](auto stored) {
    using S = decltype(stored);
    if (integer) {
      Gather<S>(file, layout, ranges, columns, integers.data(), column_ids.data());
    } else {
      Gather<S>(file, layout, ranges, columns, doubles.data(), column_ids.data());
    }
  });
  Block block;
  block.rows = row_count;
  block.columns = columns;
  block.nnz = static_cast<std::int64_t>(total);
  block.values_integer = integer;
  block.outer = selected.data();
  block.inner = column_ids.data();
  block.values = integer ? static_cast<const void*>(integers.data())
                         : static_cast<const void*>(doubles.data());
  const auto bytes = WriteBlock(target, block, integer);
  std::copy(selected.begin(), selected.end(), written_outer);
  return bytes;
}

}  // namespace meds_tab::npz
