/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 * See: https://github.com/pytorch/pytorch/blob/main/torch/csrc/profiler/combined_traceback.h
 *      https://github.com/pytorch/pytorch/blob/main/torch/csrc/profiler/unwind/unwind.h
 */

#pragma once
#include <c10/macros/Export.h>
#include <string>
#include <vector>

// copied from torch/csrc/profiler/combined_traceback.h
// and         torch/csrc/profiler/unwind/unwind.h

namespace torch {
namespace unwind {
// gather current stack, relatively fast.
// gets faster once the cache of program counter locations is warm.
TORCH_API std::vector<void*> unwind();

struct Frame {
  std::string filename;
  std::string funcname;
  uint64_t lineno;
};

// note: symbolize is really slow
// it will launch an addr2line process that has to parse dwarf
// information from the libraries that frames point into.
// Callers should first batch up all the unique void* pointers
// across a number of unwind states and make a single call to
// symbolize.
TORCH_API std::vector<Frame> symbolize(const std::vector<void*>& frames);

struct Stats {
  size_t hits = 0;
  size_t misses = 0;
  size_t unsupported = 0;
  size_t resets = 0;
};
Stats stats();

} // namespace unwind
} // namespace torch

#include <torch/csrc/jit/runtime/interpreter.h>
namespace torch {

// struct that holds the result of symbolizing multiple tracebacks
// each traceback is a list of indices into all_frames
// (lots of Frames get duplicated across traces)
struct TORCH_API SymbolizedTracebacks {
  std::vector<unwind::Frame> all_frames;
  // index into all_frames, so that
  // it is possible to dedupe frame objects in
  // construction of python objects
  std::vector<std::vector<uint64_t>> tracebacks;
};

struct TORCH_API CapturedTraceback : public c10::GatheredContext {
  struct PyFrame {
    void* code; // PyCodeObject*, but python headers not present
    int lasti;
  };

  static std::shared_ptr<CapturedTraceback> gather(
      bool python,
      bool script,
      bool cpp);
  CapturedTraceback() = default;
  CapturedTraceback(const CapturedTraceback&) = delete;
  CapturedTraceback& operator=(const CapturedTraceback&) = delete;
  ~CapturedTraceback();
  struct Python {
    virtual std::vector<PyFrame> gather() = 0;
    virtual void release(std::vector<PyFrame>& frames) = 0;
    virtual void appendSymbolized(
        const std::vector<PyFrame>& to_symbolize,
        SymbolizedTracebacks& st) = 0;
    virtual ~Python() = default;
    Python* next_ = nullptr;
  };
  // called once by each python interpreter to
  // register python stack recording functionality
  // p cannot be deleted once added.
  static void addPythonUnwinder(Python* p);

 private:
  std::vector<PyFrame> frames_;
  std::vector<void*> cpp_frames_;
  std::vector<jit::StackEntry> script_frames_;
  friend TORCH_API SymbolizedTracebacks
  symbolize(const std::vector<CapturedTraceback*>& to_symbolize);

  // non-owning reference to one of the immortal Python* objects
  // registered above.
  Python* python_ = nullptr;
};

TORCH_API SymbolizedTracebacks
symbolize(const std::vector<CapturedTraceback*>& to_symbolize);

} // namespace torch
