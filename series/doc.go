// Package series provides immutable, nullable, typed columns and the
// operations that act on them.
//
// Series is the typed column value used both independently and inside a
// dataframe. Operations align series positionally and require equal lengths
// when they combine rows. Series immutability is shallow: outer storage is not
// exposed for mutation, but references contained in element values remain
// shared.
//
// Type-changing operations such as [Series.Map], [Series.Map2], and
// [Series.Reduce] are generic methods and require Go 1.27 or newer.
package series

import "errors"

// ErrLengthMismatch is returned when a constructor receives positional inputs
// with different lengths. Operations whose signatures cannot return errors
// panic on length mismatch instead.
var ErrLengthMismatch = errors.New("series length mismatch")
