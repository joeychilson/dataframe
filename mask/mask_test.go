package mask

import (
	"runtime"
	"slices"
	"strconv"
	"testing"
)

func TestNew_CopiesSelections(t *testing.T) {
	for _, length := range []int{0, 1, 63, 64, 65, 127, 128} {
		t.Run(strconv.Itoa(length), func(t *testing.T) {
			selected := make([]bool, length)
			wantCount := 0
			for i := range selected {
				selected[i] = i%7 == 0 || i == length-1
				if selected[i] {
					wantCount++
				}
			}

			m := New(selected)
			if m.Len() != length {
				t.Fatalf("Len() = %d, want %d", m.Len(), length)
			}
			if m.Count() != wantCount {
				t.Fatalf("Count() = %d, want %d", m.Count(), wantCount)
			}
			if m.Any() != (wantCount != 0) {
				t.Fatalf("Any() = %t, want %t", m.Any(), wantCount != 0)
			}
			if m.All() != (wantCount == length) {
				t.Fatalf("All() = %t, want %t", m.All(), wantCount == length)
			}
			for i, want := range selected {
				if got := m.At(i); got != want {
					t.Fatalf("At(%d) = %t, want %t", i, got, want)
				}
			}

			if length != 0 {
				selected[0] = false
				if !m.At(0) {
					t.Fatal("changing the input changed the Mask")
				}
			}
		})
	}
}

func TestNewFunc_CallsSelectorInRowOrder(t *testing.T) {
	var calls []int
	m := NewFunc(65, func(i int) bool {
		calls = append(calls, i)
		return i == 0 || i == 64
	})

	if len(calls) != 65 {
		t.Fatalf("selected called %d times, want 65", len(calls))
	}
	for i, call := range calls {
		if call != i {
			t.Fatalf("selected call %d received row %d", i, call)
		}
	}
	if rows := slices.Collect(m.Rows()); !slices.Equal(rows, []int{0, 64}) {
		t.Fatalf("NewFunc() rows = %v, want [0 64]", rows)
	}
}

func TestAllAndNone_ConstructExpectedMasks(t *testing.T) {
	for _, length := range []int{0, 1, 63, 64, 65, 127, 128} {
		t.Run(strconv.Itoa(length), func(t *testing.T) {
			all := All(length)
			if all.Len() != length {
				t.Fatalf("All(%d).Len() = %d", length, all.Len())
			}
			if all.Count() != length {
				t.Fatalf("All(%d).Count() = %d", length, all.Count())
			}
			if all.Any() != (length != 0) {
				t.Fatalf("All(%d).Any() = %t", length, all.Any())
			}
			if !all.All() {
				t.Fatalf("All(%d).All() = false", length)
			}

			none := None(length)
			if none.Len() != length {
				t.Fatalf("None(%d).Len() = %d", length, none.Len())
			}
			if none.Count() != 0 {
				t.Fatalf("None(%d).Count() = %d", length, none.Count())
			}
			if none.Any() {
				t.Fatalf("None(%d).Any() = true", length)
			}
			if none.All() != (length == 0) {
				t.Fatalf("None(%d).All() = %t", length, none.All())
			}

			for i := range length {
				if !all.At(i) {
					t.Fatalf("All(%d).At(%d) = false", length, i)
				}
				if none.At(i) {
					t.Fatalf("None(%d).At(%d) = true", length, i)
				}
			}
		})
	}
}

func TestConstructorsPanicOnInvalidArguments(t *testing.T) {
	tests := []struct {
		name string
		call func()
	}{
		{name: "NewFunc rejects a negative length", call: func() { NewFunc(-1, func(int) bool { return false }) }},
		{name: "NewFunc rejects a nil selector", call: func() { NewFunc(0, nil) }},
		{name: "All rejects a negative length", call: func() { All(-1) }},
		{name: "None rejects a negative length", call: func() { None(-1) }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestMaskZeroValue_IsEmpty(t *testing.T) {
	var m Mask
	if m.Len() != 0 || m.Count() != 0 || m.Any() || !m.All() {
		t.Fatalf("zero Mask = {Len:%d Count:%d Any:%t All:%t}", m.Len(), m.Count(), m.Any(), m.All())
	}
	if rows := slices.Collect(m.Rows()); len(rows) != 0 {
		t.Fatalf("zero Mask rows = %v, want none", rows)
	}
	if complement := m.Not(); complement.Len() != 0 || !complement.All() {
		t.Fatalf("zero Mask complement = {Len:%d All:%t}", complement.Len(), complement.All())
	}
}

func TestAtPanicsOutOfRange(t *testing.T) {
	m := New([]bool{false, true})
	tests := []struct {
		name  string
		index int
	}{
		{name: "rejects a negative index", index: -1},
		{name: "rejects an index equal to length", index: m.Len()},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("At did not panic")
				}
			}()
			m.At(test.index)
		})
	}
}

func TestLogicalOperations_CombineSelections(t *testing.T) {
	leftValues := make([]bool, 65)
	rightValues := make([]bool, 65)
	for i := range leftValues {
		leftValues[i] = i%2 == 0
		rightValues[i] = i%3 == 0
	}
	left := New(leftValues)
	right := New(rightValues)

	tests := []struct {
		name   string
		result Mask
		want   func(int) bool
	}{
		{name: "And intersects both selections", result: left.And(right), want: func(i int) bool { return i%2 == 0 && i%3 == 0 }},
		{name: "Or unions either selection", result: left.Or(right), want: func(i int) bool { return i%2 == 0 || i%3 == 0 }},
		{name: "Not complements the selection", result: left.Not(), want: func(i int) bool { return i%2 != 0 }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			wantCount := 0
			if test.result.Len() != len(leftValues) {
				t.Fatalf("Len() = %d, want %d", test.result.Len(), len(leftValues))
			}
			for i := range leftValues {
				want := test.want(i)
				if want {
					wantCount++
				}
				if got := test.result.At(i); got != want {
					t.Fatalf("At(%d) = %t, want %t", i, got, want)
				}
			}
			if got := test.result.Count(); got != wantCount {
				t.Fatalf("Count() = %d, want %d", got, wantCount)
			}
		})
	}

	for i := range leftValues {
		if left.At(i) != leftValues[i] {
			t.Fatalf("operation changed left operand at row %d", i)
		}
		if right.At(i) != rightValues[i] {
			t.Fatalf("operation changed right operand at row %d", i)
		}
	}
}

func TestLogicalOperationsPanicOnLengthMismatch(t *testing.T) {
	short := None(1)
	long := None(2)
	tests := []struct {
		name string
		call func()
	}{
		{name: "And rejects mismatched lengths", call: func() { short.And(long) }},
		{name: "Or rejects mismatched lengths", call: func() { short.Or(long) }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestRows_IteratesSelectedIndexes(t *testing.T) {
	selected := make([]bool, 130)
	want := []int{0, 1, 63, 64, 65, 129}
	for _, row := range want {
		selected[row] = true
	}
	m := New(selected)

	if got := slices.Collect(m.Rows()); !slices.Equal(got, want) {
		t.Fatalf("Rows() = %v, want %v", got, want)
	}

	var firstThree []int
	m.Rows()(func(row int) bool {
		firstThree = append(firstThree, row)
		return len(firstThree) < 3
	})
	if !slices.Equal(firstThree, want[:3]) {
		t.Fatalf("Rows() before early termination = %v, want %v", firstThree, want[:3])
	}
}

func BenchmarkNewFunc(b *testing.B) {
	const length = 1 << 16
	b.ReportAllocs()
	var result Mask
	for b.Loop() {
		result = NewFunc(length, func(i int) bool { return i%4 == 0 })
	}
	runtime.KeepAlive(result)
}

func BenchmarkCount(b *testing.B) {
	selected := make([]bool, 1<<20)
	for i := range selected {
		selected[i] = i%3 == 0
	}
	m := New(selected)
	b.ReportAllocs()

	total := 0
	for b.Loop() {
		total += m.Count()
	}
	runtime.KeepAlive(total)
}

func BenchmarkAnd(b *testing.B) {
	leftValues := make([]bool, 1<<16)
	rightValues := make([]bool, len(leftValues))
	for i := range leftValues {
		leftValues[i] = i%2 == 0
		rightValues[i] = i%3 == 0
	}
	left := New(leftValues)
	right := New(rightValues)
	b.ReportAllocs()

	var result Mask
	for b.Loop() {
		result = left.And(right)
	}
	runtime.KeepAlive(result)
}

func BenchmarkRows(b *testing.B) {
	const length = 1 << 16
	sparseValues := make([]bool, length)
	for i := 0; i < length; i += 64 {
		sparseValues[i] = true
	}
	benchmarks := []struct {
		name string
		mask Mask
	}{
		{name: "dense", mask: All(length)},
		{name: "sparse", mask: New(sparseValues)},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			total := 0
			for b.Loop() {
				for row := range benchmark.mask.Rows() {
					total += row
				}
			}
			runtime.KeepAlive(total)
		})
	}
}
