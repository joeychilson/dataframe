package mask

import (
	"iter"
	"math/bits"
)

const bitsPerWord = 64

// Mask is an immutable, two-valued selection with one bit per row. Its zero
// value is an empty Mask.
type Mask struct {
	bits []uint64
	n    int
}

// New returns a Mask containing the selections in selected. The input is
// copied.
func New(selected []bool) Mask {
	m := None(len(selected))
	for i, selected := range selected {
		if selected {
			m.bits[i/bitsPerWord] |= uint64(1) << (i % bitsPerWord)
		}
	}
	return m
}

// NewFunc returns a Mask spanning n rows for which selected reports whether
// each row is selected. It calls selected once per row in ascending order and
// panics when n is negative.
func NewFunc(n int, selected func(int) bool) Mask {
	m := None(n)
	for i := range n {
		if selected(i) {
			m.bits[i/bitsPerWord] |= uint64(1) << (i % bitsPerWord)
		}
	}
	return m
}

// All returns a Mask selecting all n rows. It panics when n is negative.
func All(n int) Mask {
	m := None(n)
	for i := range m.bits {
		m.bits[i] = ^uint64(0)
	}
	if remainder := n % bitsPerWord; remainder != 0 {
		m.bits[len(m.bits)-1] = (uint64(1) << remainder) - 1
	}
	return m
}

// None returns a Mask selecting no rows out of n. It panics when n is
// negative.
func None(n int) Mask {
	if n < 0 {
		panic("mask: negative length")
	}

	words := n / bitsPerWord
	if n%bitsPerWord != 0 {
		words++
	}
	return Mask{bits: make([]uint64, words), n: n}
}

// Len returns the number of rows spanned by m.
func (m Mask) Len() int {
	return m.n
}

// At reports whether row i is selected. It panics when i is out of range.
func (m Mask) At(i int) bool {
	if i < 0 || i >= m.n {
		panic("mask: index out of range")
	}
	return m.bits[i/bitsPerWord]&(uint64(1)<<(i%bitsPerWord)) != 0
}

// Count returns the number of selected rows.
func (m Mask) Count() int {
	count := 0
	for _, word := range m.bits {
		count += bits.OnesCount64(word)
	}
	return count
}

// Any reports whether at least one row is selected.
func (m Mask) Any() bool {
	for _, word := range m.bits {
		if word != 0 {
			return true
		}
	}
	return false
}

// All reports whether every row is selected. It reports true for an empty
// Mask.
func (m Mask) All() bool {
	fullWords := m.n / bitsPerWord
	for _, word := range m.bits[:fullWords] {
		if word != ^uint64(0) {
			return false
		}
	}

	remainder := m.n % bitsPerWord
	if remainder == 0 {
		return true
	}
	return m.bits[fullWords] == (uint64(1)<<remainder)-1
}

// And returns the intersection of m and other. It panics on length mismatch.
func (m Mask) And(other Mask) Mask {
	if m.n != other.n {
		panic("mask: length mismatch")
	}

	result := None(m.n)
	for i, word := range m.bits {
		result.bits[i] = word & other.bits[i]
	}
	return result
}

// Or returns the union of m and other. It panics on length mismatch.
func (m Mask) Or(other Mask) Mask {
	if m.n != other.n {
		panic("mask: length mismatch")
	}

	result := None(m.n)
	for i, word := range m.bits {
		result.bits[i] = word | other.bits[i]
	}
	return result
}

// Not returns the complement of m.
func (m Mask) Not() Mask {
	result := None(m.n)
	for i, word := range m.bits {
		result.bits[i] = ^word
	}
	if remainder := m.n % bitsPerWord; remainder != 0 {
		result.bits[len(result.bits)-1] &= (uint64(1) << remainder) - 1
	}
	return result
}

// Rows iterates selected row indexes in ascending order.
func (m Mask) Rows() iter.Seq[int] {
	return func(yield func(int) bool) {
		for wordIndex, word := range m.bits {
			for word != 0 {
				bit := bits.TrailingZeros64(word)
				if !yield(wordIndex*bitsPerWord + bit) {
					return
				}
				word &= word - 1
			}
		}
	}
}
