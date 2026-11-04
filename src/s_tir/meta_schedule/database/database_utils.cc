/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/string.h>

#include <iomanip>
#include <sstream>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

void JSONDumps(Any json_obj, std::ostringstream& os) {
  if (json_obj == nullptr) {
    os << "null";
  } else if (auto opt_int_imm = json_obj.try_cast<IntImm>()) {
    IntImm int_imm = *std::move(opt_int_imm);
    PrimType int_ty = int_imm.ty();
    if (int_ty.MatchesElementType(DLDataTypeCode::kDLBool, 8) && !int_ty.IsScalableVector() &&
        !int_ty.IsFixedLengthVector()) {
      if (int_imm->value) {
        os << "true";
      } else {
        os << "false";
      }
    } else {
      os << int_imm->value;
    }
  } else if (auto opt_float_imm = json_obj.try_cast<FloatImm>()) {
    FloatImm float_imm = *std::move(opt_float_imm);
    os << std::setprecision(20) << float_imm->value;
  } else if (auto opt_str = json_obj.as<ffi::String>()) {
    os << ffi::EscapeStringJSON(*opt_str);
  } else if (const auto* array = json_obj.as<ffi::ArrayObj>()) {
    os << "[";
    int n = array->size();
    for (int i = 0; i < n; ++i) {
      if (i != 0) {
        os << ",";
      }
      JSONDumps(array->at(i), os);
    }
    os << "]";
  } else if (const auto* dict = json_obj.as<ffi::MapObj>()) {
    int n = dict->size();
    std::vector<std::pair<ffi::String, ffi::Any>> key_values;
    key_values.reserve(n);
    for (const auto& kv : *dict) {
      if (auto key = kv.first.try_cast<ffi::String>()) {
        key_values.emplace_back(key.value(), kv.second);
      } else {
        TVM_FFI_THROW(TypeError) << "Only string keys are supported in JSON dumps, but got: "
                                 << kv.first.GetTypeKey();
      }
    }
    std::sort(key_values.begin(), key_values.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    os << "{";
    for (int i = 0; i < n; ++i) {
      const auto& kv = key_values[i];
      if (i != 0) {
        os << ",";
      }
      os << ffi::EscapeStringJSON(kv.first);
      os << ":";
      JSONDumps(kv.second, os);
    }
    os << "}";
  } else if (json_obj.as<tirx::IndexMapObj>()) {
    JSONDumps(ffi::String(ffi::json::Stringify(
                  ffi::ToJSONGraph(json_obj, ffi::json::Object{{"tvm_version", TVM_VERSION}}),
                  /*indent=*/2)),
              os);
  } else {
    TVM_FFI_THROW(TypeError) << "Unsupported type in JSON object: " << json_obj.GetTypeKey();
  }
}

std::string JSONDumps(Any json_obj) {
  std::ostringstream os;
  JSONDumps(json_obj, os);
  return os.str();
}

class JSONTokenizer {
 public:
  enum class TokenType : int32_t {
    kEOF = 0,          // end of file
    kNull = 1,         // null
    kTrue = 2,         // true
    kFalse = 3,        // false
    kLeftSquare = 4,   // [
    kRightSquare = 5,  // ]
    kLeftCurly = 6,    // {
    kRightCurly = 7,   // }
    kComma = 8,        // ,
    kColon = 9,        // :
    kInteger = 10,     // integers
    kFloat = 11,       // floating point numbers
    kString = 12,      // string
  };

  struct Token {
    TokenType type;
    Any value{nullptr};
  };

  explicit JSONTokenizer(const char* st, const char* ed) : cur_(st), end_(ed) {}

  Token Next() {
    for (; cur_ != end_ && std::isspace(*cur_); ++cur_) {
    }
    if (cur_ == end_) return Token{TokenType::kEOF};
    if (NextLeftSquare()) return Token{TokenType::kLeftSquare};
    if (NextRightSquare()) return Token{TokenType::kRightSquare};
    if (NextLeftCurly()) return Token{TokenType::kLeftCurly};
    if (NextRightCurly()) return Token{TokenType::kRightCurly};
    if (NextComma()) return Token{TokenType::kComma};
    if (NextColon()) return Token{TokenType::kColon};
    if (NextNull()) return Token{TokenType::kNull};
    if (NextTrue()) return Token{TokenType::kTrue};
    if (NextFalse()) return Token{TokenType::kFalse};
    Token token;
    if (NextString(&token)) return token;
    if (NextNumber(&token)) return token;
    TVM_FFI_THROW(ValueError) << "Cannot tokenize: " << std::string(cur_, end_);
    throw;
  }

 private:
  bool NextLeftSquare() { return NextLiteral('['); }
  bool NextRightSquare() { return NextLiteral(']'); }
  bool NextLeftCurly() { return NextLiteral('{'); }
  bool NextRightCurly() { return NextLiteral('}'); }
  bool NextComma() { return NextLiteral(','); }
  bool NextColon() { return NextLiteral(':'); }
  bool NextNull() { return NextLiteral("null", 4); }
  bool NextTrue() { return NextLiteral("true", 4); }
  bool NextFalse() { return NextLiteral("false", 5); }

  bool NextNumber(Token* token) {
    bool is_float = false;
    const char* st = cur_;
    for (; cur_ != end_; ++cur_) {
      if (std::isdigit(*cur_) || *cur_ == '+' || *cur_ == '-') {
        continue;
      } else if (*cur_ == '.' || *cur_ == 'e' || *cur_ == 'E') {
        is_float = true;
      } else {
        break;
      }
    }
    if (st == cur_) {
      return false;
    }
    std::string to_parse(st, cur_);
    if (!is_float) {
      try {
        *token = Token{TokenType::kInteger, std::stoll(to_parse)};
      } catch (const std::invalid_argument& e) {
        LOG(WARNING) << "ValueError: Invalid argument to std::stoll: " << to_parse
                     << ". Details: " << e.what() << ". Switching to std::stod now.";
        is_float = true;
      } catch (const std::out_of_range& e) {
        LOG(WARNING) << "ValueError: Out-of-range for std::stoll: " << to_parse
                     << ". Details: " << e.what() << ". Switching to std::stod now.";
        is_float = true;
      }
    }
    if (is_float) {
      try {
        *token = Token{TokenType::kFloat, std::stod(to_parse)};
      } catch (const std::invalid_argument& e) {
        LOG(INFO) << "ValueError: Invalid argument to std::stod: " << to_parse
                  << ". Details: " << e.what();
      } catch (const std::out_of_range& e) {
        LOG(INFO) << "ValueError: Out-of-range for std::stod: " << to_parse
                  << ". Details: " << e.what();
      }
    }
    return true;
  }

  bool NextString(Token* token) {
    if (cur_ == end_ || *cur_ != '"') return false;
    const char* start = cur_++;
    for (; cur_ != end_; ++cur_) {
      if (*cur_ == '"') {
        ++cur_;
        // Decode using the same JSON rules as EscapeStringJSON, including Unicode escapes.
        *token = Token{TokenType::kString, ffi::json::Parse(std::string(start, cur_))};
        return true;
      }
      if (*cur_ == '\\') {
        ++cur_;
        if (cur_ == end_) break;
      }
    }
    TVM_FFI_THROW(ValueError) << "Unexpected end of string";
    return false;
  }

  bool NextLiteral(char c) {
    if (cur_ != end_ && *cur_ == c) {
      ++cur_;
      return true;
    }
    return false;
  }

  bool NextLiteral(const char* str, int len) {
    if (cur_ + len <= end_ && std::strncmp(cur_, str, len) == 0) {
      cur_ += len;
      return true;
    }
    return false;
  }
  /*! \brief The current pointer */
  const char* cur_;
  /*! \brief End of the string */
  const char* end_;

  friend class JSONParser;
};

class JSONParser {
 public:
  using TokenType = JSONTokenizer::TokenType;
  using Token = JSONTokenizer::Token;

  explicit JSONParser(const char* st, const char* ed) : tokenizer_(st, ed) {}

  Any Get() {
    Token token = tokenizer_.Next();
    if (token.type == TokenType::kEOF) {
      return ffi::ObjectRef(nullptr);
    }
    return ParseObject(std::move(token));
  }

 private:
  Any ParseObject(Token token) {
    switch (token.type) {
      case TokenType::kNull:
        return Any(nullptr);
      case TokenType::kTrue:
        return Any(true);
      case TokenType::kFalse:
        return Any(false);
      case TokenType::kLeftSquare:
        return ParseArray();
      case TokenType::kLeftCurly:
        return ParseDict();
      case TokenType::kString:
      case TokenType::kInteger:
      case TokenType::kFloat:
        return token.value;
      case TokenType::kRightSquare:
        TVM_FFI_THROW(ValueError) << "Unexpected token: ]";
      case TokenType::kRightCurly:
        TVM_FFI_THROW(ValueError) << "Unexpected token: }";
      case TokenType::kComma:
        TVM_FFI_THROW(ValueError) << "Unexpected token: ,";
      case TokenType::kColon:
        TVM_FFI_THROW(ValueError) << "Unexpected token: :";
      case TokenType::kEOF:
        TVM_FFI_THROW(ValueError) << "Unexpected EOF";
      default:
        throw;
    }
  }

  ffi::Array<Any> ParseArray() {
    bool is_first = true;
    ffi::Array<Any> results;
    for (;;) {
      Token token;
      if (is_first) {
        is_first = false;
        token = Token{TokenType::kComma};
      } else {
        token = tokenizer_.Next();
      }
      // Three cases overall:
      // - Case 1. 1 token: "]"
      // - Case 2. 2 tokens: ",", "]"
      // - Case 3. 2 tokens: ",", "obj"
      if (token.type == TokenType::kRightSquare) {  // Case 1
        break;
      } else if (token.type == TokenType::kComma) {
        token = tokenizer_.Next();
        if (token.type == TokenType::kRightSquare) {  // Case 2
          break;
        }
        // Case 3
        results.push_back(ParseObject(std::move(token)));
        continue;
      } else {
        TVM_FFI_THROW(ValueError) << "Unexpected token before: " << tokenizer_.cur_;
      }
    }
    return results;
  }

  ffi::Map<ffi::String, ffi::Any> ParseDict() {
    bool is_first = true;
    ffi::Map<ffi::String, ffi::Any> results;
    for (;;) {
      Token token;
      if (is_first) {
        is_first = false;
        token = Token{TokenType::kComma};
      } else {
        token = tokenizer_.Next();
      }
      // Three cases overall:
      // - Case 1. 1 token: "}"
      // - Case 2. 2 tokens: ",", "}"
      // - Case 3. 2 tokens: ",", "key", ":", "value"
      if (token.type == TokenType::kRightCurly) {  // Case 1
        break;
      } else if (token.type == TokenType::kComma) {
        token = tokenizer_.Next();
        if (token.type == TokenType::kRightCurly) {  // Case 2
          break;
        }
        // Case 3
        Any key = ParseObject(std::move(token));
        TVM_FFI_CHECK(key.as<ffi::String>(), ValueError)
            << "key must be a string, but gets: " << key;
        token = tokenizer_.Next();
        TVM_FFI_CHECK(token.type == TokenType::kColon, ValueError)
            << "Unexpected token before: " << tokenizer_.cur_;
        Any value = ParseObject(tokenizer_.Next());
        results.Set(key.as_or_throw<ffi::String>(), value);
        continue;
      } else {
        TVM_FFI_THROW(ValueError) << "Unexpected token before: " << tokenizer_.cur_;
      }
    }
    return results;
  }

  JSONTokenizer tokenizer_;
};

Any JSONLoads(std::string str) {
  const char* st = str.c_str();
  const char* ed = st + str.length();
  return JSONParser(st, ed).Get();
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
