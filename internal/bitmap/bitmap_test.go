package bitmap

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"testing"
)

func TestInitializedEmptyBitmap_DiffersFromZeroValue(t *testing.T) {
	var zero Bitmap
	if zero.Initialized() {
		t.Fatal("zero Bitmap is initialized")
	}
	if zero.Clone().Initialized() {
		t.Fatal("cloning a zero Bitmap initialized it")
	}
	empty := New(0)
	if !empty.Initialized() || empty.Len() != 0 || empty.Count() != 0 || !empty.All() {
		t.Fatalf("empty Bitmap = {Initialized:%t Len:%d Count:%d All:%t}", empty.Initialized(), empty.Len(), empty.Count(), empty.All())
	}
	if !empty.Clone().Initialized() {
		t.Fatal("cloning an initialized empty Bitmap lost initialization")
	}
}

func TestConstructorsAndMutation_PreserveBitValues(t *testing.T) {
	var calls []int
	bitmap := NewFunc(65, func(i int) bool {
		calls = append(calls, i)
		return i == 0 || i == 64
	})
	if len(calls) != 65 {
		t.Fatalf("NewFunc call count = %d, want 65", len(calls))
	}
	for i, call := range calls {
		if call != i {
			t.Fatalf("NewFunc call %d received %d", i, call)
		}
	}
	if got, want := slices.Collect(bitmap.Rows()), []int{0, 64}; !slices.Equal(got, want) {
		t.Fatalf("NewFunc rows = %v, want %v", got, want)
	}

	bitmap.Set(64, false)
	bitmap.Set(1, true)
	if got := slices.Collect(bitmap.Rows()); !slices.Equal(got, []int{0, 1}) {
		t.Fatalf("rows after Set = %v, want [0 1]", got)
	}

	filled := Filled(65)
	words := filled.words
	if len(words) != 2 || words[0] != ^uint64(0) || words[1] != 1 {
		t.Fatalf("Filled(65) words = %#v", words)
	}
}

func TestBitmapOperations_PanicOnInvalidArguments(t *testing.T) {
	bitmap := New(2)
	tests := []struct {
		name string
		call func()
	}{
		{name: "New rejects a negative length", call: func() { New(-1) }},
		{name: "NewFunc rejects a negative length", call: func() { NewFunc(-1, func(int) bool { return false }) }},
		{name: "NewFunc rejects a nil selector", call: func() { NewFunc(0, nil) }},
		{name: "Filled rejects a negative length", call: func() { Filled(-1) }},
		{name: "At rejects a negative index", call: func() { bitmap.At(-1) }},
		{name: "At rejects an index equal to length", call: func() { bitmap.At(bitmap.Len()) }},
		{name: "Set rejects a negative index", call: func() { bitmap.Set(-1, true) }},
		{name: "Set rejects an index equal to length", call: func() { bitmap.Set(bitmap.Len(), true) }},
		{name: "Slice rejects a negative bound", call: func() { bitmap.Slice(-1, 1) }},
		{name: "Slice rejects reversed bounds", call: func() { bitmap.Slice(2, 1) }},
		{name: "Slice rejects an end past length", call: func() { bitmap.Slice(0, 3) }},
		{name: "Copy rejects a negative start", call: func() { bitmap.Copy(-1, New(1)) }},
		{name: "Copy rejects a source past length", call: func() { bitmap.Copy(2, New(1)) }},
		{name: "And rejects mismatched lengths", call: func() { bitmap.And(New(1)) }},
		{name: "Or rejects mismatched lengths", call: func() { bitmap.Or(New(1)) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			assertPanics(t, test.call)
		})
	}
}

func TestSliceSharesUnalignedStorage(t *testing.T) {
	values := make([]bool, 130)
	for _, i := range []int{0, 1, 63, 64, 65, 129} {
		values[i] = true
	}
	source := FromBools(values)
	view := source.Slice(1, 66)
	if got, want := view.Bools(), values[1:66]; !slices.Equal(got, want) {
		t.Fatalf("Slice bits = %v, want %v", got, want)
	}
	if got, want := slices.Collect(view.Rows()), []int{0, 62, 63, 64}; !slices.Equal(got, want) {
		t.Fatalf("Slice rows = %v, want %v", got, want)
	}
	if got := slices.Collect(view.UnsetRows()); len(got) != view.Len()-4 || got[0] != 1 || got[len(got)-1] != 61 {
		t.Fatalf("Slice unset rows = %v", got)
	}
	if view.Count() != 4 || !view.Any() || view.All() {
		t.Fatalf("Slice summary = {Count:%d Any:%t All:%t}", view.Count(), view.Any(), view.All())
	}

	source.Set(2, true)
	if !view.At(1) {
		t.Fatal("Slice did not share source storage")
	}
	clone := view.Clone()
	source.Set(2, false)
	if !clone.At(1) {
		t.Fatal("Clone shared source storage")
	}
	if view.At(1) {
		t.Fatal("Slice stopped reflecting source storage")
	}
}

func TestCopy_ReplacesOnlyTheDestinationRange(t *testing.T) {
	destination := Filled(10)
	source := FromBools([]bool{true, false, true, false}).Slice(1, 4)
	destination.Copy(3, source)
	want := []bool{true, true, true, false, true, false, true, true, true, true}
	if got := destination.Bools(); !slices.Equal(got, want) {
		t.Fatalf("Copy bits = %v, want %v", got, want)
	}
}

func TestAlignedCopyPreservesAdjacentBits(t *testing.T) {
	values := make([]bool, 130)
	values[0] = true
	values[64] = true
	values[65] = true
	source := FromBools(values).Slice(0, 65)
	destination := Filled(130)
	destination.Copy(64, source)

	want := make([]bool, 130)
	for i := range 64 {
		want[i] = true
	}
	want[64] = true
	want[128] = true
	want[129] = true
	if got := destination.Bools(); !slices.Equal(got, want) {
		t.Fatalf("Copy bits = %v, want %v", got, want)
	}
}

func TestCopyOverlap_BehavesLikeTemporaryCopy(t *testing.T) {
	lengths := []int{0, 1, 63, 64, 65, 127, 128, 129}
	tests := []struct {
		name                 string
		destinationViewStart int
		destinationStart     int
		sourceStart          int
	}{
		{name: "aligned forward overlap copies through temporary storage", destinationStart: 64},
		{name: "aligned backward overlap copies through temporary storage", sourceStart: 64},
		{name: "unaligned forward overlap copies through temporary storage", destinationViewStart: 1, destinationStart: 1, sourceStart: 1},
		{name: "unaligned backward overlap copies through temporary storage", destinationViewStart: 1, sourceStart: 2},
		{name: "unaligned forward cross-word overlap copies through temporary storage", destinationStart: 66, sourceStart: 1},
		{name: "unaligned backward cross-word overlap copies through temporary storage", destinationStart: 1, sourceStart: 66},
		{name: "unaligned forward word view copies through temporary storage", destinationViewStart: 65},
		{name: "unaligned backward word view copies through temporary storage", destinationViewStart: 64, sourceStart: 65},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, length := range lengths {
				values := make([]bool, 260)
				for i := range values {
					values[i] = i%3 == 0 || i%11 == 0
				}
				want := slices.Clone(values)
				destinationStart := test.destinationViewStart + test.destinationStart
				copy(want[destinationStart:destinationStart+length], want[test.sourceStart:test.sourceStart+length])

				bitmap := FromBools(values)
				destination := bitmap.Slice(test.destinationViewStart, bitmap.Len())
				source := bitmap.Slice(test.sourceStart, test.sourceStart+length)
				destination.Copy(test.destinationStart, source)
				if got := bitmap.Bools(); !slices.Equal(got, want) {
					t.Fatalf("Copy length %d differs from built-in copy", length)
				}
			}
		})
	}
}

func TestLogicalOperationsOnUnalignedViews(t *testing.T) {
	leftValues := make([]bool, 131)
	rightValues := make([]bool, 131)
	for i := range leftValues {
		leftValues[i] = i%2 == 0
		rightValues[i] = i%3 == 0
	}
	left := FromBools(leftValues).Slice(1, 130)
	right := FromBools(rightValues).Slice(1, 130)

	and := left.And(right)
	or := left.Or(right)
	not := left.Not()
	for i := range left.Len() {
		leftValue := leftValues[i+1]
		rightValue := rightValues[i+1]
		if got, want := and.At(i), leftValue && rightValue; got != want {
			t.Fatalf("And.At(%d) = %t, want %t", i, got, want)
		}
		if got, want := or.At(i), leftValue || rightValue; got != want {
			t.Fatalf("Or.At(%d) = %t, want %t", i, got, want)
		}
		if got, want := not.At(i), !leftValue; got != want {
			t.Fatalf("Not.At(%d) = %t, want %t", i, got, want)
		}
	}
}

func TestAlignedOrClearsTrailingStorage(t *testing.T) {
	left := Filled(130).Slice(0, 65)
	result := left.Or(New(65))
	words := result.words
	if !result.All() || len(words) != 2 || words[0] != ^uint64(0) || words[1] != 1 {
		t.Fatalf("Or result = {All:%t Words:%#v}", result.All(), words)
	}
}

func TestWordCountDoesNotOverflow(t *testing.T) {
	want := math.MaxInt / bitsPerWord
	if math.MaxInt%bitsPerWord != 0 {
		want++
	}
	if got := wordCount(math.MaxInt); got != want {
		t.Fatalf("wordCount(math.MaxInt) = %d, want %d", got, want)
	}
	if got := wordCount(0); got != 0 {
		t.Fatalf("wordCount(0) = %d, want 0", got)
	}
}

func FuzzViewAndCopyAgainstBoolSlice(f *testing.F) {
	boundary := make([]byte, 260)
	for i := range boundary {
		boundary[i] = byte(i)
	}
	for _, sourceOffset := range []uint16{0, 1, 63} {
		for _, destinationOffset := range []uint16{0, 1, 63} {
			for _, length := range []uint16{0, 1, 63, 64, 65} {
				f.Add(boundary, uint16(0), uint16(260), sourceOffset, destinationOffset, uint16(0), length)
			}
		}
	}
	f.Add(boundary, uint16(1), uint16(130), uint16(0), uint16(64), uint16(0), uint16(129))
	f.Add(boundary, uint16(63), uint16(193), uint16(65), uint16(1), uint16(1), uint16(128))
	f.Add([]byte{}, uint16(0), uint16(0), uint16(0), uint16(0), uint16(0), uint16(0))

	f.Fuzz(func(t *testing.T, data []byte, rawStart, rawEnd, rawSource, rawDestination, rawView, rawLength uint16) {
		data = data[:min(len(data), 260)]
		values := make([]bool, len(data))
		for i, value := range data {
			values[i] = value&1 != 0
		}

		start := int(rawStart) % (len(values) + 1)
		end := int(rawEnd) % (len(values) + 1)
		if start > end {
			start, end = end, start
		}
		view := FromBools(values).Slice(start, end)
		wantView := values[start:end]
		if got := view.Bools(); !slices.Equal(got, wantView) {
			t.Fatalf("Slice(%d, %d) = %v, want %v", start, end, got, wantView)
		}
		var wantRows []int
		for i, value := range wantView {
			if value {
				wantRows = append(wantRows, i)
			}
		}
		if got := slices.Collect(view.Rows()); !slices.Equal(got, wantRows) {
			t.Fatalf("Slice(%d, %d).Rows() = %v, want %v", start, end, got, wantRows)
		}
		if view.Count() != len(wantRows) || view.Any() != (len(wantRows) > 0) || view.All() != (len(wantRows) == len(wantView)) {
			t.Fatalf("Slice(%d, %d) summary = Count:%d Any:%t All:%t", start, end, view.Count(), view.Any(), view.All())
		}
		if got := view.Clone().Bools(); !slices.Equal(got, wantView) {
			t.Fatalf("Slice(%d, %d).Clone() = %v, want %v", start, end, got, wantView)
		}

		sourceStart := int(rawSource) % (len(values) + 1)
		destinationStart := int(rawDestination) % (len(values) + 1)
		viewStart := int(rawView) % (destinationStart + 1)
		length := int(rawLength) % (min(len(values)-sourceStart, len(values)-destinationStart) + 1)
		want := slices.Clone(values)
		copy(want[destinationStart:destinationStart+length], want[sourceStart:sourceStart+length])
		bitmap := FromBools(values)
		destination := bitmap.Slice(viewStart, bitmap.Len())
		source := bitmap.Slice(sourceStart, sourceStart+length)
		destination.Copy(destinationStart-viewStart, source)
		if got := bitmap.Bools(); !slices.Equal(got, want) {
			t.Fatalf("Copy source [%d:%d] to %d through view %d = %v, want %v", sourceStart, sourceStart+length, destinationStart, viewStart, got, want)
		}
	})
}

func assertPanics(t *testing.T, call func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("call did not panic")
		}
	}()
	call()
}

func BenchmarkCopy(b *testing.B) {
	const size = 1 << 16
	for _, sourceOffset := range []int{0, 1, 63} {
		for _, destinationOffset := range []int{0, 1, 63} {
			name := fmt.Sprintf("SourceOffset=%d/DestinationOffset=%d", sourceOffset, destinationOffset)
			b.Run(name, func(b *testing.B) {
				source := Filled(size+sourceOffset).Slice(sourceOffset, size+sourceOffset)
				b.Run("Optimized", func(b *testing.B) {
					destination := New(size + destinationOffset)
					b.ReportAllocs()
					for b.Loop() {
						destination.Copy(destinationOffset, source)
					}
					runtime.KeepAlive(destination)
				})
				b.Run("Reference", func(b *testing.B) {
					destination := New(size + destinationOffset)
					b.ReportAllocs()
					for b.Loop() {
						for i := range source.Len() {
							destination.Set(destinationOffset+i, source.At(i))
						}
					}
					runtime.KeepAlive(destination)
				})
			})
		}
	}
}

func BenchmarkReadAndLogicalOperations(b *testing.B) {
	const size = 1 << 16
	leftBase := NewFunc(size+1, func(i int) bool { return i%3 == 0 })
	rightBase := NewFunc(size+1, func(i int) bool { return i%5 != 0 })
	for _, offset := range []int{0, 1} {
		left := leftBase.Slice(offset, offset+size)
		right := rightBase.Slice(offset, offset+size)
		name := fmt.Sprintf("Offset=%d", offset)
		anyBase := New(size + offset)
		anyBase.Set(size+offset-1, true)
		anyInput := anyBase.Slice(offset, offset+size)
		allBase := Filled(size + offset)
		allBase.Set(size+offset-1, false)
		allInput := allBase.Slice(offset, offset+size)
		for _, operation := range []struct {
			name      string
			optimized func() bool
			reference func() bool
		}{
			{name: "Any", optimized: anyInput.Any, reference: func() bool {
				for i := range anyInput.Len() {
					if anyInput.At(i) {
						return true
					}
				}
				return false
			}},
			{name: "All", optimized: allInput.All, reference: func() bool {
				for i := range allInput.Len() {
					if !allInput.At(i) {
						return false
					}
				}
				return true
			}},
		} {
			b.Run(name+"/"+operation.name, func(b *testing.B) {
				b.Run("Optimized", func(b *testing.B) {
					b.ReportAllocs()
					var result bool
					for b.Loop() {
						result = operation.optimized()
					}
					runtime.KeepAlive(result)
				})
				b.Run("Reference", func(b *testing.B) {
					b.ReportAllocs()
					var result bool
					for b.Loop() {
						result = operation.reference()
					}
					runtime.KeepAlive(result)
				})
			})
		}
		b.Run(name+"/Count", func(b *testing.B) {
			b.Run("Optimized", func(b *testing.B) {
				b.ReportAllocs()
				total := 0
				for b.Loop() {
					total += left.Count()
				}
				runtime.KeepAlive(total)
			})
			b.Run("Reference", func(b *testing.B) {
				b.ReportAllocs()
				total := 0
				for b.Loop() {
					for i := range left.Len() {
						if left.At(i) {
							total++
						}
					}
				}
				runtime.KeepAlive(total)
			})
		})
		b.Run(name+"/Rows", func(b *testing.B) {
			b.Run("Optimized", func(b *testing.B) {
				b.ReportAllocs()
				total := 0
				for b.Loop() {
					for row := range left.Rows() {
						total += row
					}
				}
				runtime.KeepAlive(total)
			})
			b.Run("Reference", func(b *testing.B) {
				b.ReportAllocs()
				total := 0
				for b.Loop() {
					for i := range left.Len() {
						if left.At(i) {
							total += i
						}
					}
				}
				runtime.KeepAlive(total)
			})
		})
		for _, operation := range []struct {
			name      string
			optimized func() Bitmap
			reference func() Bitmap
		}{
			{name: "Clone", optimized: left.Clone, reference: func() Bitmap {
				result := New(left.Len())
				for i := range left.Len() {
					result.Set(i, left.At(i))
				}
				return result
			}},
			{name: "And", optimized: func() Bitmap { return left.And(right) }, reference: func() Bitmap {
				result := New(left.Len())
				for i := range left.Len() {
					result.Set(i, left.At(i) && right.At(i))
				}
				return result
			}},
			{name: "Or", optimized: func() Bitmap { return left.Or(right) }, reference: func() Bitmap {
				result := New(left.Len())
				for i := range left.Len() {
					result.Set(i, left.At(i) || right.At(i))
				}
				return result
			}},
			{name: "Not", optimized: left.Not, reference: func() Bitmap {
				result := New(left.Len())
				for i := range left.Len() {
					result.Set(i, !left.At(i))
				}
				return result
			}},
		} {
			b.Run(name+"/"+operation.name, func(b *testing.B) {
				b.Run("Optimized", func(b *testing.B) {
					b.ReportAllocs()
					var result Bitmap
					for b.Loop() {
						result = operation.optimized()
					}
					runtime.KeepAlive(result)
				})
				b.Run("Reference", func(b *testing.B) {
					b.ReportAllocs()
					var result Bitmap
					for b.Loop() {
						result = operation.reference()
					}
					runtime.KeepAlive(result)
				})
			})
		}
	}
}

func BenchmarkStorage(b *testing.B) {
	const size = 1 << 16
	b.Run("Optimized", func(b *testing.B) {
		b.ReportAllocs()
		var result Bitmap
		for b.Loop() {
			result = New(size)
		}
		runtime.KeepAlive(result)
	})
	b.Run("Reference", func(b *testing.B) {
		b.ReportAllocs()
		var result []bool
		for b.Loop() {
			result = make([]bool, size)
		}
		runtime.KeepAlive(result)
	})
}
