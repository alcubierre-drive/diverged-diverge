#pragma once

#ifdef __cplusplus
extern "C" {
#endif

inline int integer_length(int value) {
    int l = !value;
    while (value) {
        l++;
        value/=10;
    }
    return l;
}

#ifdef __cplusplus
}
#endif
