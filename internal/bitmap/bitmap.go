// Package bitmap provides the shared bit-packed representation used for row
// validity and selection inside dataframe.
package bitmap

import (
	"iter"
	"math/bits"
)

// Bitmap is an immutable-by-convention sequence of bits. Its zero value is an
// uninitialized empty bitmap; New(0) is initialized so callers can preserve
// schema state for empty data. Slice returns a shared view.
type Bitmap struct {
	words  []uint64
	offset int
	length int
}

// New returns an initialized bitmap containing n cleared bits. It panics when
// n is negative.
func New(n int) Bitmap {
	if n < 0 {
		panic("bitmap: negative length")
	}
	return Bitmap{words: make([]uint64, wordCount(n)), length: n}
}

// FromBools returns an initialized bitmap containing values.
func FromBools(values []bool) Bitmap {
	result := New(len(values))
	for wordIndex := range result.words {
		start := wordIndex * bitsPerWord
		end := min(start+bitsPerWord, len(values))
		word := uint64(0)
		for bit, value := range values[start:end] {
			if value {
				word |= uint64(1) << bit
			}
		}
		result.words[wordIndex] = word
	}
	return result
}

// NewFunc returns an initialized bitmap containing n bits for which selected
// reports true. It calls selected once per position in ascending order. NewFunc
// panics when n is negative or selected is nil.
func NewFunc(n int, selected func(int) bool) Bitmap {
	result := New(n)
	if selected == nil {
		panic("bitmap: nil selection function")
	}
	for i := range n {
		if selected(i) {
			result.words[i/bitsPerWord] |= uint64(1) << (i % bitsPerWord)
		}
	}
	return result
}

// Filled returns an initialized bitmap containing n set bits. It panics when n
// is negative.
func Filled(n int) Bitmap {
	result := New(n)
	for i := range result.words {
		result.words[i] = ^uint64(0)
	}
	if remainder := n % bitsPerWord; remainder != 0 {
		result.words[len(result.words)-1] = lowMask(remainder)
	}
	return result
}

// Initialized reports whether the bitmap was constructed, including when its
// length is zero.
func (b Bitmap) Initialized() bool {
	return b.words != nil
}

// Len returns the number of bits.
func (b Bitmap) Len() int {
	return b.length
}

// At reports whether bit i is set. It panics when i is out of range.
func (b Bitmap) At(i int) bool {
	if i < 0 || i >= b.length {
		panic("bitmap: index out of range")
	}
	return b.at(i)
}

// Set changes bit i. Callers must finish constructing a bitmap before sharing
// it with immutable values. It panics when i is out of range.
func (b *Bitmap) Set(i int, value bool) {
	if i < 0 || i >= b.length {
		panic("bitmap: index out of range")
	}
	b.set(i, value)
}

// Count returns the number of set bits.
func (b Bitmap) Count() int {
	count := 0
	if b.offset == 0 {
		fullWords := b.length / bitsPerWord
		for _, word := range b.words[:fullWords] {
			count += bits.OnesCount64(word)
		}
		if remainder := b.length % bitsPerWord; remainder != 0 {
			count += bits.OnesCount64(b.words[fullWords] & lowMask(remainder))
		}
		return count
	}
	for i := range wordCount(b.length) {
		count += bits.OnesCount64(b.word(i))
	}
	return count
}

// Any reports whether at least one bit is set.
func (b Bitmap) Any() bool {
	if b.offset == 0 {
		fullWords := b.length / bitsPerWord
		for _, word := range b.words[:fullWords] {
			if word != 0 {
				return true
			}
		}
		remainder := b.length % bitsPerWord
		return remainder != 0 && b.words[fullWords]&lowMask(remainder) != 0
	}
	for i := range wordCount(b.length) {
		if b.word(i) != 0 {
			return true
		}
	}
	return false
}

// All reports whether every bit is set. It reports true for an empty bitmap.
func (b Bitmap) All() bool {
	if b.offset == 0 {
		fullWords := b.length / bitsPerWord
		for _, word := range b.words[:fullWords] {
			if word != ^uint64(0) {
				return false
			}
		}
		remainder := b.length % bitsPerWord
		mask := lowMask(remainder)
		return remainder == 0 || b.words[fullWords]&mask == mask
	}
	words := wordCount(b.length)
	for i := range words {
		word := b.word(i)
		remaining := b.length - i*bitsPerWord
		if remaining >= bitsPerWord {
			if word != ^uint64(0) {
				return false
			}
			continue
		}
		if word != lowMask(remaining) {
			return false
		}
	}
	return true
}

// Bools returns one boolean per bit.
func (b Bitmap) Bools() []bool {
	values := make([]bool, b.length)
	for i := range values {
		values[i] = b.at(i)
	}
	return values
}

// Clone returns an independent initialized bitmap with the same bits.
func (b Bitmap) Clone() Bitmap {
	if !b.Initialized() {
		return Bitmap{}
	}
	result := New(b.length)
	if b.offset == 0 {
		copy(result.words, b.words)
		if remainder := b.length % bitsPerWord; remainder != 0 {
			result.words[len(result.words)-1] &= lowMask(remainder)
		}
		return result
	}
	for i := range result.words {
		result.words[i] = b.word(i)
	}
	return result
}

// Slice returns bits in [start, end) as a shared view. It panics on invalid
// bounds.
func (b Bitmap) Slice(start, end int) Bitmap {
	if start < 0 || end < start || end > b.length {
		panic("bitmap: slice bounds out of range")
	}
	absoluteStart := b.offset + start
	firstWord := absoluteStart / bitsPerWord
	lastWord := (b.offset + end + bitsPerWord - 1) / bitsPerWord
	words := b.words[firstWord:lastWord:lastWord]
	return Bitmap{
		words:  words,
		offset: absoluteStart % bitsPerWord,
		length: end - start,
	}
}

// Copy copies every bit from source into b starting at start. Source and b may
// share storage; Copy behaves as if source were copied to a temporary bitmap.
// It panics when source does not fit at start.
func (b *Bitmap) Copy(start int, source Bitmap) {
	if start < 0 || start > b.length-source.length {
		panic("bitmap: copy out of range")
	}
	if source.length == 0 {
		return
	}

	destinationStart := b.offset + start
	if destinationStart%bitsPerWord == source.offset {
		destinationWord := destinationStart / bitsPerWord
		firstCount := min(source.length, bitsPerWord-source.offset)
		firstMask := lowMask(firstCount) << source.offset
		firstWord := source.words[0]
		if firstCount == source.length {
			b.words[destinationWord] = b.words[destinationWord]&^firstMask | firstWord&firstMask
			return
		}

		remaining := source.length - firstCount
		fullWords := remaining / bitsPerWord
		trailingCount := remaining % bitsPerWord
		trailingWord := uint64(0)
		if trailingCount != 0 {
			trailingWord = source.words[1+fullWords]
		}
		copy(b.words[destinationWord+1:], source.words[1:1+fullWords])
		b.words[destinationWord] = b.words[destinationWord]&^firstMask | firstWord&firstMask
		if trailingCount != 0 {
			mask := lowMask(trailingCount)
			word := &b.words[destinationWord+1+fullWords]
			*word = *word&^mask | trailingWord&mask
		}
		return
	}
	b.copyUnaligned(start, source)
}

func (b *Bitmap) copyUnaligned(start int, source Bitmap) {
	backward := requiresBackwardCopy(*b, start, source)
	remaining := source.length
	for remaining > 0 {
		count := min(bitsPerWord, remaining)
		position := source.length - remaining
		if backward {
			if count = remaining % bitsPerWord; count == 0 {
				count = bitsPerWord
			}
			position = remaining - count
		}

		value := source.word(position / bitsPerWord)
		absolute := b.offset + start + position
		wordIndex := absolute / bitsPerWord
		shift := absolute % bitsPerWord
		firstCount := min(count, bitsPerWord-shift)
		if firstCount == bitsPerWord {
			b.words[wordIndex] = value
		} else {
			mask := lowMask(firstCount) << shift
			b.words[wordIndex] = b.words[wordIndex]&^mask | value<<shift&mask
		}
		if secondCount := count - firstCount; secondCount > 0 {
			mask := lowMask(secondCount)
			b.words[wordIndex+1] = b.words[wordIndex+1]&^mask | value>>firstCount&mask
		}

		if backward {
			remaining = position
		} else {
			remaining -= count
		}
	}
}

// And returns the intersection of b and other. It panics on length mismatch.
func (b Bitmap) And(other Bitmap) Bitmap {
	if b.length != other.length {
		panic("bitmap: length mismatch")
	}
	result := New(b.length)
	if b.offset == 0 && other.offset == 0 {
		for i := range result.words {
			result.words[i] = b.words[i] & other.words[i]
		}
		if remainder := b.length % bitsPerWord; remainder != 0 {
			result.words[len(result.words)-1] &= lowMask(remainder)
		}
		return result
	}
	for i := range result.words {
		result.words[i] = b.word(i) & other.word(i)
	}
	return result
}

// Or returns the union of b and other. It panics on length mismatch.
func (b Bitmap) Or(other Bitmap) Bitmap {
	if b.length != other.length {
		panic("bitmap: length mismatch")
	}
	result := New(b.length)
	if b.offset == 0 && other.offset == 0 {
		for i := range result.words {
			result.words[i] = b.words[i] | other.words[i]
		}
		if remainder := b.length % bitsPerWord; remainder != 0 {
			result.words[len(result.words)-1] &= lowMask(remainder)
		}
		return result
	}
	for i := range result.words {
		result.words[i] = b.word(i) | other.word(i)
	}
	return result
}

// Not returns the complement of b.
func (b Bitmap) Not() Bitmap {
	result := New(b.length)
	if b.offset == 0 {
		for i, word := range b.words {
			result.words[i] = ^word
		}
		if remainder := b.length % bitsPerWord; remainder != 0 {
			result.words[len(result.words)-1] &= lowMask(remainder)
		}
		return result
	}
	for i := range result.words {
		result.words[i] = ^b.word(i)
	}
	if remainder := b.length % bitsPerWord; remainder != 0 {
		result.words[len(result.words)-1] &= lowMask(remainder)
	}
	return result
}

// Rows iterates set bit indexes in ascending order.
func (b Bitmap) Rows() iter.Seq[int] {
	return func(yield func(int) bool) {
		if b.offset == 0 {
			for wordIndex, word := range b.words {
				if wordIndex == len(b.words)-1 {
					if remainder := b.length % bitsPerWord; remainder != 0 {
						word &= lowMask(remainder)
					}
				}
				for word != 0 {
					bit := bits.TrailingZeros64(word)
					if !yield(wordIndex*bitsPerWord + bit) {
						return
					}
					word &= word - 1
				}
			}
			return
		}
		for wordIndex := range wordCount(b.length) {
			word := b.word(wordIndex)
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

// UnsetRows iterates cleared bit indexes in ascending order.
func (b Bitmap) UnsetRows() iter.Seq[int] {
	return func(yield func(int) bool) {
		for wordIndex := range wordCount(b.length) {
			word := ^b.word(wordIndex)
			if remaining := b.length - wordIndex*bitsPerWord; remaining < bitsPerWord {
				word &= lowMask(remaining)
			}
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

const bitsPerWord = 64

func wordCount(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + (n-1)/bitsPerWord
}

func lowMask(n int) uint64 {
	return uint64(1)<<n - 1
}

func (b Bitmap) at(i int) bool {
	absolute := b.offset + i
	return b.words[absolute/bitsPerWord]&(uint64(1)<<(absolute%bitsPerWord)) != 0
}

func (b *Bitmap) set(i int, value bool) {
	absolute := b.offset + i
	word := &b.words[absolute/bitsPerWord]
	bit := uint64(1) << (absolute % bitsPerWord)
	if value {
		*word |= bit
	} else {
		*word &^= bit
	}
}

func (b Bitmap) word(i int) uint64 {
	start := b.offset + i*bitsPerWord
	wordIndex := start / bitsPerWord
	shift := start % bitsPerWord
	word := b.words[wordIndex] >> shift
	if shift != 0 && wordIndex+1 < len(b.words) {
		word |= b.words[wordIndex+1] << (bitsPerWord - shift)
	}
	if remaining := b.length - i*bitsPerWord; remaining < bitsPerWord {
		word &= lowMask(remaining)
	}
	return word
}

func requiresBackwardCopy(destination Bitmap, start int, source Bitmap) bool {
	if source.length == 0 {
		return false
	}
	destinationStart := destination.offset + start
	destinationWord := &destination.words[destinationStart/bitsPerWord]
	for i := range source.words {
		if destinationWord == &source.words[i] {
			relativeStart := i*bitsPerWord + destinationStart%bitsPerWord
			return relativeStart > source.offset && relativeStart < source.offset+source.length
		}
	}
	return false
}
