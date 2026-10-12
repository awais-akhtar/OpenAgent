#ifndef OA_NATIVE_EXPORTS_H
#define OA_NATIVE_EXPORTS_H
#if defined(_WIN32)
#define OA_API __declspec(dllexport)
#define OA_CALL __cdecl
#else
#define OA_API __attribute__((visibility("default")))
#define OA_CALL
#endif
#endif
