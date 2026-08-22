package reduce

import "testing"

type cells[T any] struct {
	values   []T
	validity []bool
}

func (c cells[T]) Len() int {
	return len(c.values)
}

func (c cells[T]) At(row int) (T, bool) {
	return c.values[row], c.validity == nil || c.validity[row]
}

func TestReductions(t *testing.T) {
	values := cells[int]{
		values:   []int{8, 3, 5, 2},
		validity: []bool{true, false, true, true},
	}
	rows := []int{3, 1, 2}

	if got, present := Sum(values, rows); got != 7 || !present {
		t.Fatalf("Sum() = (%d, %v), want (7, true)", got, present)
	}
	if got, present := Mean(values, rows); got != 3.5 || !present {
		t.Fatalf("Mean() = (%v, %v), want (3.5, true)", got, present)
	}
	if got, present := Min(values, rows); got != 2 || !present {
		t.Fatalf("Min() = (%d, %v), want (2, true)", got, present)
	}
	if got, present := Max(values, rows); got != 5 || !present {
		t.Fatalf("Max() = (%d, %v), want (5, true)", got, present)
	}
	if got, present := FirstPresent(values, rows); got != 2 || !present {
		t.Fatalf("FirstPresent() = (%d, %v), want (2, true)", got, present)
	}
	if got, present := LastPresent(values, rows); got != 5 || !present {
		t.Fatalf("LastPresent() = (%d, %v), want (5, true)", got, present)
	}
}

func TestReductionsSelectAllAndEmpty(t *testing.T) {
	values := cells[int]{values: []int{4, 9}}
	if got, present := Sum(values, nil); got != 13 || !present {
		t.Fatalf("Sum(nil) = (%d, %v), want (13, true)", got, present)
	}
	if _, present := Sum(values, []int{}); present {
		t.Fatal("Sum(empty) reported a present value")
	}
	if _, present := FirstPresent(values, []int{}); present {
		t.Fatal("FirstPresent(empty) reported a present value")
	}
	if _, present := LastPresent(values, []int{}); present {
		t.Fatal("LastPresent(empty) reported a present value")
	}
}
