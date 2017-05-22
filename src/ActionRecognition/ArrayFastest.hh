/*  -*- c++ -*-  */
#ifndef ARRAY_FASTEST_H
#define ARRAY_FASTEST_H

#include <algorithm>
#include <assert.h>

namespace utils {
  //****************************************************************************
  // Class Template for a Generic Resizable N Dimensional Array       (for N>=2)
  // By Giovanni Bavestrelli                 Copyright 1999 Giovanni Bavestrelli
  // Any feedback is welcome, you can contact me at gbavestrelli@yahoo.com
  //
  // http://www.cuj.com/documents/s=8032/cuj0012bavestre/
  //
  // This is the full implementation of my array, up to the new C++ standard.
  // It uses partial specialization, and it will not work with Visual C++.
  // If you use Visual C++, take the other version of the classes.
  //
  // This version uses Andrei Alexandrescu's idea of inheriting
  // RefArray<T,N> from RefArray<T,N-1> so that RefArray<T,N>::operator [] can
  // return a reference to a RefArray instead of a new RefArray, for improved
  // efficiency.
  //
  // Here I made a further optimization:
  // I keep a RefArray<T,N-1> as private data member inside my Array<T,N> class
  // so that in operator [] I return a reference to this RefArray instead of
  // creating a new one. All indexing is done through the same RefArray object.
  // This is not thread safe, and potentially dengerous, but it's very fast.
  // I suggest you use one of the other versions in multithreaded applications,
  // and anyway test your application very well if you dare to use this version.
  //
  // TM: resize() will check first if the dimesions are the same and call
  //  doResize()
  //    if re-size is really needed.
  //
  //    All unsigned int changed to size_t
  //
  //    Further optimization by removing asserts. Use '#define DEBUG_USE_ASSERT'
  //    to make use of asserts.
  //
  //****************************************************************************

  // Forward declaration needed for friend declarations
  template<typename T, size_t N>
  class Array;

  //============================================================================
  // lasses for passing a typesafe vector of dimensions to the Array constructor
  //============================================================================

  // Class that encapsulates a const size_t (&)[N]
  template<size_t N>
  class ArraySize {
#ifndef ARRAYSIZE_NOT_TYPE_SAFE
    typedef const size_t (&UIntArrayN)[N];
#else
    typedef const size_t *UIntArrayN;
#endif

    size_t m_Dimensions[N];

    ArraySize(const size_t ( &Dimensions)[N - 1], size_t dim) {
      std::copy(&Dimensions[0], &Dimensions[N - 1], m_Dimensions);
      m_Dimensions[N - 1] = dim;
    }

  public:

    ArraySize < N + 1 > operator()(size_t dim) {
      return ArraySize < N + 1 > (m_Dimensions, dim);
    }

    operator UIntArrayN() const {return m_Dimensions;}

    friend class ArraySizes;
    friend class ArraySize < N - 1 >;
  };

  // Starting point to build a const size_t (&)[N] on the fly
  class ArraySizes {
    size_t m_Dimensions[1];

  public:

    explicit ArraySizes(size_t dim) {
      m_Dimensions[0] = dim;
    }

    ArraySize<2> operator()(size_t dim) {
      return ArraySize<2>(m_Dimensions, dim);
    }
  };

  //----------------------------------------------------------------------------
  // Class Template for N Dimensional SubArrays within an Array
  //----------------------------------------------------------------------------

  template<typename T, size_t N>
  class RefArray : public RefArray < T, N - 1 > {
  public:

    // STL-like types
    typedef T value_type;
    typedef T &reference;
    typedef const T &const_reference;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T *iterator;
    typedef const T *const_iterator;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    // Give access to number of dimensions
    enum  {array_dims = N};

  private:

    size_type m_NDimensions; // Array dimensions
    size_type m_SubArrayLen; // SubArray dimensions
#if defined (__sgi)
    mutable bool m_boolDummy;
#endif
  protected:

    mutable T * m_pElements; // Point to SubArray with elements within Array

    RefArray<T, N>()
      : RefArray < T, N - 1 > (), m_NDimensions(0), m_SubArrayLen(0),
      m_pElements(NULL)
    {}

    void SetupDimensions(const size_type *pNDimensions,
                         const size_type *pSubArrayLen) {
      assert(pNDimensions && pSubArrayLen);
      assert(pNDimensions[0] > 0 && pSubArrayLen[0] > 0);

      m_NDimensions = pNDimensions[0];
      m_SubArrayLen = pSubArrayLen[0];
      m_pElements = NULL;

      RefArray < T, N - 1 > ::SetupDimensions(pNDimensions + 1,
        pSubArrayLen + 1);
    }

    void ResetDimensions() {
      m_NDimensions = 0;
      m_SubArrayLen = 0;
      m_pElements = NULL;

      RefArray < T, N - 1 > ::ResetDimensions();
    }

  public:

    RefArray < T, N - 1 > &operator[](size_type Index) {
      assert(m_pElements);
      assert(Index < m_NDimensions);
#if defined (__sgi)
      m_boolDummy = (m_pElements != NULL);
#endif
      RefArray < T, N - 1 > ::m_pElements = &m_pElements[Index * m_SubArrayLen];
      return *this;
    }

    const RefArray < T, N - 1 > &operator[](size_type Index) const {
      assert(m_pElements);
      assert(Index < m_NDimensions);
#if defined (__sgi)
      m_boolDummy = (m_pElements != NULL);
#endif
      RefArray < T, N - 1 > ::m_pElements = &m_pElements[Index * m_SubArrayLen];
      return *this;
    }

    // Return STL-like iterators
    iterator begin()       {return m_pElements;}
    const_iterator begin() const {return m_pElements;}
    iterator end()         {return m_pElements + size();}
    const_iterator end()   const {return m_pElements + size();}

    // Return size of array
    size_type size()  const {return m_NDimensions * m_SubArrayLen;}

    // Return size of subdimensions
    size_type size(size_t Dim) const {
      assert(Dim >= 1 && Dim <= N);
      if (Dim == 1) return m_NDimensions;
      else return RefArray < T, N - 1 > ::size(Dim - 1);
    }

    // Return number of dimensions
    size_t dimensions()  const {return N;}

  protected:

    // The following are protected mainly because they are not exception-safe
    // but the way they are used in the rest of the class is exception-safe

    // Copy the elements of another subarray on this one where possible
    // Where not possible, initialize them to a specified value Init
    void copy(const RefArray<T, N> &SA, const T &Init = T()) {
      size_type below = std::min(size(1), SA.size(1));
      size_type above = size(1);

      // Copy the elements we can copy
      for (size_type i = 0; i < below; ++i)
        (*this)[i].copy(SA[i], Init);

      // Reset the elements we can't copy
      for (size_type j = below; j < above; ++j)
        (*this)[j].initialize(Init);
    }

    // Reset all the elements
    void initialize(const T &Init = T()) {
      std::fill(begin(), end(), Init);
    }

    friend class Array<T, N>;
    friend class Array < T, N + 1 >;
    friend class RefArray < T, N + 1 >;
  };


  //----------------------------------------------------------------------------
  // Partial Specialization for Monodimensional SubArray within an Array
  //----------------------------------------------------------------------------

  template<typename T>
  class RefArray<T, 1> {
  public:

    // STL-like types
    typedef T value_type;
    typedef T &reference;
    typedef const T &const_reference;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T *iterator;
    typedef const T *const_iterator;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    // Give access to number of dimensions
    enum  {array_dims = 1};

  private:

    size_type m_NDimensions; // Array dimension

  protected:

    mutable T * m_pElements; // Point to elements within Array

    RefArray<T, 1>()
      : m_NDimensions(0), m_pElements(NULL)
    {}

    void SetupDimensions(const size_type *pNDimensions,
                         const size_type *pSubArrayLen) {
      assert(pNDimensions && pSubArrayLen);
      // We found the elements
      assert(pNDimensions[0] > 0 && pSubArrayLen[0] == 1);

      m_NDimensions = pNDimensions[0];
      m_pElements = NULL;
    }

    void ResetDimensions() {
      m_NDimensions = 0;
      m_pElements = NULL;
    }

  public:

    reference operator[](size_type Index) {
      assert(m_pElements);
      assert(Index < m_NDimensions);
      return m_pElements[Index];
    }

    const_reference operator[](size_type Index) const {
      assert(m_pElements);
      assert(Index < m_NDimensions);
      return m_pElements[Index];
    }

    // Return STL-like iterators
    iterator begin()       {return m_pElements;}
    const_iterator begin() const {return m_pElements;}
    iterator end()         {return m_pElements + size();}
    const_iterator end()   const {return m_pElements + size();}

    // Return size of array
    size_type size()  const {return m_NDimensions;}

    // Return size of subdimensions
    size_type size(size_t Dim) const {
      assert(Dim == 1);
      return m_NDimensions;
    }

    // Return number of dimensions
    size_t dimensions()  const {return 1;}

  protected:

    // The following are protected mainly because they are not exception-safe
    // but the way they are used in the rest of the class is exception-safe

    // Copy the elements of another subarray on this one where possible
    // Where not possible, initialize them to a specified value Init
    void copy(const RefArray<T, 1> &SA, const T &Init = T()) {
      size_type below = std::min(size(1), SA.size(1));
      size_type above = size(1);

      // Copy the elements we can copy
      for (size_type i = 0; i < below; ++i)
        m_pElements[i] = SA.m_pElements[i];

      // Reset the elements we can't copy
      for (size_type j = below; j < above; ++j)
        m_pElements[j] = Init;
    }

    // Reset all the elements
    void initialize(const T &Init = T()) {
      std::fill(begin(), end(), Init);
    }

    friend class Array<T, 1>;
    friend class Array<T, 2>;
    friend class RefArray<T, 2>;
  };


  //----------------------------------------------------------------------------
  // Class Template for a Generic Resizable N Dimensional Array
  //----------------------------------------------------------------------------

  template<typename T, size_t N>
  class Array {
  public:

    // STL-like types
    typedef T value_type;
    typedef T &reference;
    typedef const T &const_reference;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T *iterator;
    typedef const T *const_iterator;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    // Give access to number of dimensions
    enum  {array_dims = N};

  private:

    T *m_pArrayElements; // Pointer to actual array elements
    size_type m_nArrayElements; // Total number of array elements

    size_type m_NDimensions[N];  // Size of the N array dimensions
    size_type m_SubArrayLen[N];  // Size of each subarray

    RefArray < T, N - 1 > m_TheSubArray; // The only RefArray element used

  public:

    // Default constructor
    Array<T, N>()
      : m_pArrayElements(NULL), m_nArrayElements(0) {
      std::fill(m_NDimensions, m_NDimensions + N, 0);
      std::fill(m_SubArrayLen, m_SubArrayLen + N, 0);
    }

    // This takes an array of N values representing the size of the N dimensions
    // You must remove the explicit keyword if you use MetroWerks CodeWarrior
#ifndef ARRAYSIZE_NOT_TYPE_SAFE
    explicit Array<T, N>(const size_t ( &Dimensions)[N], const T &Init = T())
#else
    explicit Array<T, N>(const size_t Dimensions[], const T &Init = T())
#endif
      : m_pArrayElements(NULL), m_nArrayElements(0) {
      std::fill(m_NDimensions, m_NDimensions + N, 0);
      std::fill(m_SubArrayLen, m_SubArrayLen + N, 0);

      resize(Dimensions, Init);
    }

    // Copy constructor
    Array<T, N>(const Array<T, N> &A)
      : m_pArrayElements(NULL), m_nArrayElements(0) {
      std::fill(m_NDimensions, m_NDimensions + N, 0);
      std::fill(m_SubArrayLen, m_SubArrayLen + N, 0);

      Array<T, N> Temp;
      if (!A.empty() && Temp.resize(A.m_NDimensions))
        std::copy(A.begin(), A.end(), Temp.begin());
      swap(Temp);
    }

    // Destructor
    ~Array<T, N>()
    {
      delete[] m_pArrayElements;
    }

    // DANGEROUS OPTIMIZATION:
    // The '&' sign in the following two [] operators is the dangerous but
    // fast optimization. All indexing is done through the same RefArray object.
    // So the following two [] operators are very fast but somewhat dangerous
    // as they always return a reference to the same indexing object.
    // The result will certainly not be thread safe, and it could possibly even
    // lead to index the wrong element in some strange situations.
    // I tried it a bit and it always worked fine, so if you need maximum speed
    // and test it well in a single threaded application, you might consider it.
    // But you do it at your own risk, you have been warned.

    // Indexing Array
    RefArray < T, N - 1 > &operator[](size_type Index) {
      assert(m_pArrayElements);
      assert(Index < m_NDimensions[0]);
      m_TheSubArray.m_pElements = &m_pArrayElements[Index * m_SubArrayLen[0]];
      return m_TheSubArray;
    }

    // Indexing Constant Array
    const RefArray < T, N - 1 > &operator[](size_type Index) const {
      assert(m_pArrayElements);
      assert(Index < m_NDimensions[0]);
      m_TheSubArray.m_pElements = &m_pArrayElements[Index * m_SubArrayLen[0]];
      return m_TheSubArray;
    }

    // Return RefArray referencing entire Array
    RefArray<T, N> GetRefArray() {
      assert(m_pArrayElements);
      RefArray<T, N> RA;
      RA.SetupDimensions(m_NDimensions, m_SubArrayLen);
      RA.m_pElements = m_pArrayElements;
      return RA;
    }

    // Return constant RefArray referencing entire Array
    const RefArray<T, N> GetRefArray() const {
      assert(m_pArrayElements);
      RefArray<T, N> RA;
      RA.SetupDimensions(m_NDimensions, m_SubArrayLen);
      RA.m_pElements = m_pArrayElements;
      return RA;
    }

    // Set the size of each array dimension
    // We check if we really have to re-size (TM-2002-02-04)
#ifndef ARRAYSIZE_NOT_TYPE_SAFE
    bool resize(const size_t ( &Dimensions)[N],
                const T &Init = T(), bool PreserveElems = false)
#else
    bool resize(const size_t Dimensions[],
                const T &Init = T(), bool PreserveElems = false)
#endif
    {
      for (int i = 0; (size_t)i < N; ++i) {
        if (Dimensions[i] < 1)
          return false;   // Check that no dimension was zero
        if (Dimensions[i] != m_NDimensions[i])
          return doResize(Dimensions, Init, PreserveElems); // We need a re-size
      }

      if (!PreserveElems)
        initialize(Init);

      return true;
    }

    // Set the size of each array dimension
#ifndef ARRAYSIZE_NOT_TYPE_SAFE
    bool doResize(const size_t ( &Dimensions)[N],
                  const T &Init = T(), bool PreserveElems = false)
#else
    bool doResize(const size_t Dimensions[],
                  const T &Init = T(), bool PreserveElems = false)
#endif

    {
      Array<T, N> Temp;

      // Calculate all the information you need to use the array
      Temp.m_nArrayElements = 1;
      for (int i = 0; (size_t)i < N; ++i) {
        if (Dimensions[i] == 0)
          return false;   // Check that no dimension was zero
        Temp.m_nArrayElements *= Dimensions[i];
        Temp.m_NDimensions[i] = Dimensions[i];
        Temp.m_SubArrayLen[i] = 1;
        for (int k = N - 1; k > i; k--)
          Temp.m_SubArrayLen[i] *= Dimensions[k];
      }

      // Allocate new elements, let exception propagate
      Temp.m_pArrayElements = new T[Temp.m_nArrayElements];

      // Some compilers might not throw exception if allocation fails
      //assert(Temp.m_pArrayElements);
      if (!Temp.m_pArrayElements)
        return false;

      // Setup the RefArray needed for indexing
      Temp.m_TheSubArray.SetupDimensions(Temp.m_NDimensions + 1,
        Temp.m_SubArrayLen + 1);

      // Copy the elements from the previous array if requested
      if (PreserveElems && !empty())
        Temp.copy(*this, Init);
      // Otherwise initialize them to the specified value
      else
        Temp.initialize(Init);

      // Now swap this object with the temporary
      swap(Temp);

      return true;
    }

    // Delete the complete Array
    void clear() {
      delete[] m_pArrayElements;
      m_pArrayElements = NULL;
      m_nArrayElements = 0;

      std::fill(m_NDimensions, m_NDimensions + N, 0);
      std::fill(m_SubArrayLen, m_SubArrayLen + N, 0);

      // Reset the RefArray needed for indexing
      m_TheSubArray.ResetDimensions();
    }

    // Assignment operator
    Array<T, N> &operator=(const Array<T, N> &A) {
      if (&A != this) { // For efficiency
        Array<T, N> Temp(A);
        swap(Temp);
      }
      return *this;
    }

    // Return STL-like iterators
    iterator begin()       {return m_pArrayElements;}
    const_iterator begin() const {return m_pArrayElements;}
    iterator end()         {return m_pArrayElements + m_nArrayElements;}
    const_iterator end()   const {return m_pArrayElements + m_nArrayElements;}

    // Some more STL-like size members
    size_type size()       const {return m_nArrayElements;}

    // Return the size of each dimension, 1 to N
    size_type size(size_t Dim) const {
      assert(Dim >= 1 && Dim <= N);
      return m_NDimensions[Dim - 1];
    }

    // Say if the array is empty
    bool empty()           const {return m_nArrayElements == 0;}

    // Return number of dimensions
    size_t dimensions()  const {return N;}

    // Swap this array with another, a'la STL
    void swap(Array<T, N> &A) {
      std::swap(m_pArrayElements, A.m_pArrayElements);
      std::swap(m_nArrayElements, A.m_nArrayElements);

      std::swap_ranges(m_NDimensions, m_NDimensions + N, A.m_NDimensions);
      std::swap_ranges(m_SubArrayLen, m_SubArrayLen + N, A.m_SubArrayLen);

      std::swap(m_TheSubArray, A.m_TheSubArray);
    }

  protected:

    // The following are protected mainly because they are not exception-safe
    // but the way they are used in the rest of the class is exception-safe

    // Copy the elements of another array on this one where possible
    // Where not possible, initialize them to a specified value Init
    void copy(const Array<T, N> &A, const T &Init = T()) {
      size_type below = std::min(size(1), A.size(1));
      size_type above = size(1);

      // Copy the elements we can copy
      for (size_type i = 0; i < below; ++i)
        (*this)[i].copy(A[i], Init);

      // Reset the elements we can't copy
      for (size_type j = below; j < above; ++j)
        (*this)[j].initialize(Init);
    }

    // Initialize all the array elements
    void initialize(const T &Init = T()) {
      std::fill(begin(), end(), Init);
    }

    // Prefer non-member operator ==, but it needs to be a friend
    template<typename TT, size_t NN>
    friend bool operator==(const Array<TT, NN> &A, const Array<TT, NN> &B);
  };

  // Test for equality between two arrays
  template<typename T, size_t N>
  bool operator==(const Array<T, N> &A, const Array<T, N> &B);
  template<typename T, size_t N>
  inline bool operator==(const Array<T, N> &A, const Array<T, N> &B) {
    return std::equal(A.m_NDimensions, A.m_NDimensions + N, B.m_NDimensions) &&
           std::equal(A.begin(), A.end(), B.begin());
  }

  // Test for inequality between two arrays
  template<typename T, size_t N>
  bool operator!=(const Array<T, N> &A, const Array<T, N> &B);
  template<typename T, size_t N>
  inline bool operator!=(const Array<T, N> &A, const Array<T, N> &B) {
    return !(A == B);
  }


  // Not implemented, meaningless to have 0 dimensions
  template<typename T>
  class Array<T, 0> {};


  // Not implemented, use std::vector for one dimensional arrays
  template<typename T>
  class Array<T, 1> {};
}
#endif
