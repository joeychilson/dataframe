package mask

import (
	"iter"
	"math/bits"

	"github.com/joeychilson/dataframe/internal/bitmap"
)

const bitsPerWord = 64

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
// panics when n is negative.
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
	if i < 0 || i >= m.Len() {
		panic("mask: index out of range")
	}
	words := bitmap.Bitmap(m).AlignedWords()
	return words[i/bitsPerWord]&(uint64(1)<<(i%bitsPerWord)) != 0
}

// Count returns the number of selected rows.
func (m Mask) Count() int {
	count := 0
	for _, word := range bitmap.Bitmap(m).AlignedWords() {
		count += bits.OnesCount64(word)
	}
	return count
}

// Any reports whether at least one row is selected.
func (m Mask) Any() bool {
	for _, word := range bitmap.Bitmap(m).AlignedWords() {
		if word != 0 {
			return true
		}
	}
	return false
}

// All reports whether every row is selected. It reports true for an empty
// Mask.
func (m Mask) All() bool {
	words := bitmap.Bitmap(m).AlignedWords()
	fullWords := m.Len() / bitsPerWord
	for _, word := range words[:fullWords] {
		if word != ^uint64(0) {
			return false
		}
	}

	remainder := m.Len() % bitsPerWord
	return remainder == 0 || words[fullWords] == uint64(1)<<remainder-1
}

// And returns the intersection of m and other. It panics on length mismatch.
func (m Mask) And(other Mask) Mask {
	if m.Len() != other.Len() {
		panic("mask: length mismatch")
	}
	leftWords := bitmap.Bitmap(m).AlignedWords()
	rightWords := bitmap.Bitmap(other).AlignedWords()
	result := bitmap.New(m.Len())
	resultWords := result.AlignedWords()
	for i, word := range leftWords {
		resultWords[i] = word & rightWords[i]
	}
	return Mask(result)
}

// Or returns the union of m and other. It panics on length mismatch.
func (m Mask) Or(other Mask) Mask {
	if m.Len() != other.Len() {
		panic("mask: length mismatch")
	}
	leftWords := bitmap.Bitmap(m).AlignedWords()
	rightWords := bitmap.Bitmap(other).AlignedWords()
	result := bitmap.New(m.Len())
	resultWords := result.AlignedWords()
	for i, word := range leftWords {
		resultWords[i] = word | rightWords[i]
	}
	return Mask(result)
}

// Not returns the complement of m.
func (m Mask) Not() Mask {
	result := bitmap.New(m.Len())
	resultWords := result.AlignedWords()
	for i, word := range bitmap.Bitmap(m).AlignedWords() {
		resultWords[i] = ^word
	}
	if remainder := m.Len() % bitsPerWord; remainder != 0 {
		resultWords[len(resultWords)-1] &= uint64(1)<<remainder - 1
	}
	return Mask(result)
}

// Rows iterates selected row indexes in ascending order.
func (m Mask) Rows() iter.Seq[int] {
	words := bitmap.Bitmap(m).AlignedWords()
	return func(yield func(int) bool) {
		for wordIndex, word := range words {
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
