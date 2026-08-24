package reduce

import (
	"math"
	"slices"
	"testing"
)

func TestReductions_IterateValues(t *testing.T) {
	values := slices.All([]int{8, 5, 2})
	if got, present := Sum(values); got != 15 || !present {
		t.Fatalf("Sum() = (%d, %v), want (15, true)", got, present)
	}
	if got, present := Mean(values); got != 5 || !present {
		t.Fatalf("Mean() = (%v, %v), want (5, true)", got, present)
	}
	if got, present := Min(values); got != 2 || !present {
		t.Fatalf("Min() = (%d, %v), want (2, true)", got, present)
	}
	if got, present := Max(values); got != 8 || !present {
		t.Fatalf("Max() = (%d, %v), want (8, true)", got, present)
	}
}

func TestReductions_ReportEmptySequence(t *testing.T) {
	empty := slices.All([]int(nil))
	if _, present := Sum(empty); present {
		t.Fatal("Sum(empty) reported a value")
	}
	if _, present := Mean(empty); present {
		t.Fatal("Mean(empty) reported a value")
	}
	if _, present := Min(empty); present {
		t.Fatal("Min(empty) reported a value")
	}
	if _, present := Max(empty); present {
		t.Fatal("Max(empty) reported a value")
	}
}

func TestMeanAvoidsIntermediateOverflow(t *testing.T) {
	maximums := slices.All([]float64{math.MaxFloat64, math.MaxFloat64})
	if got, present := Mean(maximums); got != math.MaxFloat64 || !present {
		t.Fatalf("Mean() = (%v, %v), want (%v, true)", got, present, math.MaxFloat64)
	}
	opposite := slices.All([]float64{-math.MaxFloat64, math.MaxFloat64})
	if got, present := Mean(opposite); got != 0 || !present {
		t.Fatalf("Mean(-MaxFloat64, MaxFloat64) = (%v, %v), want (0, true)", got, present)
	}
	infinite := slices.All([]float64{math.Inf(1), math.Inf(1)})
	if got, present := Mean(infinite); !math.IsInf(got, 1) || !present {
		t.Fatalf("Mean(+Inf, +Inf) = (%v, %v), want (+Inf, true)", got, present)
	}
}
