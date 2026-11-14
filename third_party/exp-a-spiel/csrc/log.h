#pragma once

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
// coda-r: isatty / fileno come from <io.h> under MSVC
#ifdef _WIN32
#include <io.h>
#define EAS_ISATTY(f) _isatty(_fileno(f))
#else
#include <unistd.h>
#define EAS_ISATTY(f) isatty(fileno(f))
#endif

#ifdef NDEBUG
#define COMPILATION_MODE "RELEASE (NDEBUG)"
#else
#define COMPILATION_MODE "DEBUG"
#endif

namespace {
void _PrintMsSinceEpoch() {
  long int ms; // Milliseconds
  time_t s;    // Seconds
  struct timespec spec;

  // coda-r: portable C++17 replacement for POSIX clock_gettime(CLOCK_REALTIME, ...)
  timespec_get(&spec, TIME_UTC);

  s = spec.tv_sec;
  ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
  if (ms > 999)
    ++s, ms = 0;

  printf("[%" PRIdMAX ".%03ld", (intmax_t)s, ms);
}
} // namespace

#ifndef __FILE_NAME__
#define __FILE_NAME__                                                          \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define INFO(...)                                                              \
  {                                                                            \
    _PrintMsSinceEpoch();                                                      \
    printf("|>%s] [%s:%03d] ",                                                 \
           EAS_ISATTY(stdout) ? "\033[32mINFO\033[0m" : "INFO",                \
           __FILE_NAME__, __LINE__);                                           \
    printf(__VA_ARGS__);                                                       \
    printf("\n");                                                              \
  }

#define FATAL(...)                                                             \
  {                                                                            \
    _PrintMsSinceEpoch();                                                      \
    printf("|%s] [%s:%03d] ",                                                  \
           EAS_ISATTY(stdout) ? "\033[1m\033[31mFATAL\033[0m" : "FATAL",       \
           __FILE_NAME__, __LINE__);                                           \
    printf(__VA_ARGS__);                                                       \
    printf("\n");                                                              \
    std::abort();                                                              \
  }

#define CHECK(CONDITION, ...)                                                  \
  {                                                                            \
    if (!(CONDITION)) [[unlikely]]                                             \
      FATAL(__VA_ARGS__)                                                       \
  }
