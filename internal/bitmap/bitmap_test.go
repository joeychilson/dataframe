package bitmap

import (
	"runtime"
	"slices"
	"testing"
)

func TestInitializedEmptyBitmap(t *testing.T) {
	var zero Bitmap
	if zero.Initialized() {
		t.Fatal("zero Bitmap is initialized")
	}
	empty := New(0)
	if !empty.Initialized() || empty.Len() != 0 || empty.Count() != 0 || !empty.All() {
		t.Fatalf("empty Bitmap = {Initialized:%t Len:%d Count:%d All:%t}", empty.Initialized(), empty.Len(), empty.Count(), empty.All())
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
	for i := 0; i < left.Len(); i++ {
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

func TestCopy(t *testing.T) {
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

func TestCopyOverlap(t *testing.T) {
	lengths := []int{0, 1, 63, 64, 65, 127, 128, 129}
	tests := []struct {
		name                 string
		destinationViewStart int
		destinationStart     int
		sourceStart          int
	}{
		{name: "aligned forward", destinationStart: 64},
		{name: "aligned backward", sourceStart: 64},
		{name: "unaligned forward", destinationViewStart: 1, destinationStart: 1, sourceStart: 1},
		{name: "unaligned backward", destinationViewStart: 1, sourceStart: 2},
		{name: "unaligned forward across words", destinationStart: 66, sourceStart: 1},
		{name: "unaligned backward across words", destinationStart: 1, sourceStart: 66},
		{name: "unaligned forward word view", destinationViewStart: 65},
		{name: "unaligned backward word view", destinationViewStart: 64, sourceStart: 65},
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

func BenchmarkCopy(b *testing.B) {
	const size = 1 << 16

	b.Run("Aligned", func(b *testing.B) {
		source := Filled(size)
		destination := New(size)
		b.ReportAllocs()
		for b.Loop() {
			destination.Copy(0, source)
		}
		runtime.KeepAlive(destination)
	})

	b.Run("Unaligned", func(b *testing.B) {
		source := Filled(size+1).Slice(1, size+1)
		destination := New(size + 1)
		b.ReportAllocs()
		for b.Loop() {
			destination.Copy(1, source)
		}
		runtime.KeepAlive(destination)
	})
}
