// Package mask provides immutable, bit-packed row selections.
//
// Masks are deliberately independent of dataframe and series so both packages
// can consume them without an import cycle. Series comparisons produce Mask
// values, and series.AsMask converts nullable boolean values to a Mask.
package mask

import (
	"iter"

	"github.com/joeychilson/dataframe/internal/bitmap"
)

// Mask is an immutable, two-valued selection with one bit per row. Its zero
// value is an empty Mask. Its storage is word-aligned and unused trailing bits
// are always clear.
type Mask bitmap.Bitmap

// New returns a Mask containing the selections in selected. The input is
// copied.
func New(selected []bool) Mask {
	return Mask(bitmap.FromBools(selected))
}

// NewFunc returns a Mask spanning n rows for which selected reports whether
// each row is selected. It calls selected once per row in ascending order and
// panics when n is negative or selected is nil.
func NewFunc(n int, selected func(int) bool) Mask {
	return Mask(bitmap.NewFunc(n, selected))
}

// All returns a Mask selecting all n rows. It panics when n is negative.
func All(n int) Mask {
	return Mask(bitmap.Filled(n))
}

// None returns a Mask selecting no rows out of n. It panics when n is
// negative.
func None(n int) Mask {
	return Mask(bitmap.New(n))
}

// Len returns the number of rows spanned by m.
func (m Mask) Len() int {
	return bitmap.Bitmap(m).Len()
}

// At reports whether row i is selected. It panics when i is out of range.
func (m Mask) At(i int) bool {
	return bitmap.Bitmap(m).At(i)
}

// Count returns the number of selected rows.
func (m Mask) Count() int {
	return bitmap.Bitmap(m).Count()
}

// Any reports whether at least one row is selected.
func (m Mask) Any() bool {
	return bitmap.Bitmap(m).Any()
}

// All reports whether every row is selected. It reports true for an empty
// Mask.
func (m Mask) All() bool {
	return bitmap.Bitmap(m).All()
}

// And returns the intersection of m and other. It panics on length mismatch.
func (m Mask) And(other Mask) Mask {
	return Mask(bitmap.Bitmap(m).And(bitmap.Bitmap(other)))
}

// Or returns the union of m and other. It panics on length mismatch.
func (m Mask) Or(other Mask) Mask {
	return Mask(bitmap.Bitmap(m).Or(bitmap.Bitmap(other)))
}

// Not returns the complement of m.
func (m Mask) Not() Mask {
	return Mask(bitmap.Bitmap(m).Not())
}

// Rows iterates selected row indexes in ascending order.
func (m Mask) Rows() iter.Seq[int] {
	return bitmap.Bitmap(m).Rows()
}
