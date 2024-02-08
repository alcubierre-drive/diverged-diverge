/* Copyright (C) 2024 Jonas B. Profe, Lennart Klebl; GPLv3+ (LICENSE.md) */

#pragma once
template<typename T>
inline void mcopy(T* to, const T* from, const long long unsigned size)
{memcpy((void*)to,(void*)from,size*sizeof(T));}

template<typename T>
inline void mfill_0(T* ptr, const long long unsigned size)
{std::fill(ptr,ptr+size,(T)0.0);}

template<typename T, typename D>
inline void mfill(T* ptr, const long long unsigned size, D value)
{std::fill(ptr,ptr+size,value);}

template<typename T>
inline void msort(T* ptr, const long long unsigned size)
{std::sort(ptr,ptr+size);}

