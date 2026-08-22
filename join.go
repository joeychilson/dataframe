package dataframe

import (
	"fmt"
	"hash/maphash"
	"math"

	"github.com/joeychilson/dataframe/internal/hashmap"
	"github.com/joeychilson/dataframe/series"
)

// JoinKey is an opaque typed join condition. Construct one with On, OnBy,
// Using, UsingBy, UsingColumns, or UsingColumnsBy.
//
// On and OnBy join positional Series and retain all columns from both frames.
// Using variants resolve stored columns by name and emit one coalesced key.
// Null keys never match. Comparable joins use Go ==, so NaN does not match
// itself; By variants use the supplied Hasher's equivalence relation.
type JoinKey[K any] struct {
	left        series.Series[K]
	right       series.Series[K]
	newLookup   func(int) joinLookup[K]
	leftName    string
	rightName   string
	outputName  string
	usingStored bool
}

// On joins positional comparable keys using ==. It retains every original
// column from both frames; name collisions return ErrColumnConflict.
func On[K comparable](left, right series.Series[K]) JoinKey[K] {
	return JoinKey[K]{
		left:  left,
		right: right,
		newLookup: func(capacity int) joinLookup[K] {
			return make(comparableJoinLookup[K], capacity)
		},
	}
}

// OnBy is On with a custom hash and equivalence relation. It supports
// non-comparable keys.
func OnBy[K any](left, right series.Series[K], hasher maphash.Hasher[K]) JoinKey[K] {
	var newLookup func(int) joinLookup[K]
	if hasher != nil {
		newLookup = func(capacity int) joinLookup[K] {
			return hashmap.New[K, joinMatch](hasher, capacity)
		}
	}
	return JoinKey[K]{left: left, right: right, newLookup: newLookup}
}

// Using joins same-named stored columns using == and emits one coalesced column
// under name. K must be supplied explicitly because it cannot be inferred from
// a column name.
func Using[K comparable](name string) JoinKey[K] {
	return UsingColumns[K](name, name, name)
}

// UsingBy is Using with a custom hash and equivalence relation. It supports
// non-comparable stored key columns. K must be supplied explicitly.
func UsingBy[K any](name string, hasher maphash.Hasher[K]) JoinKey[K] {
	return UsingColumnsBy(name, name, name, hasher)
}

// UsingColumns joins differently named stored columns using == and emits one
// coalesced key under outputName. K must be supplied explicitly.
func UsingColumns[K comparable](outputName, leftName, rightName string) JoinKey[K] {
	return JoinKey[K]{
		newLookup: func(capacity int) joinLookup[K] {
			return make(comparableJoinLookup[K], capacity)
		},
		leftName:    leftName,
		rightName:   rightName,
		outputName:  outputName,
		usingStored: true,
	}
}

// UsingColumnsBy is UsingColumns with a custom hash and equivalence relation.
// It supports non-comparable stored key columns. K must be supplied explicitly.
func UsingColumnsBy[K any](outputName, leftName, rightName string, hasher maphash.Hasher[K]) JoinKey[K] {
	var newLookup func(int) joinLookup[K]
	if hasher != nil {
		newLookup = func(capacity int) joinLookup[K] {
			return hashmap.New[K, joinMatch](hasher, capacity)
		}
	}
	return JoinKey[K]{
		newLookup:   newLookup,
		leftName:    leftName,
		rightName:   rightName,
		outputName:  outputName,
		usingStored: true,
	}
}

// InnerJoin returns matching row pairs. Rows are left-major, and matches for a
// left row retain right-row order. Positional key length, dynamic schema, and
// output name conflicts return errors. On variants produce the complete left
// schema followed by the complete right schema. Using variants place the
// coalesced key at the left key's position, omit the right key, and otherwise
// retain left-then-right schema order.
func (f Frame) InnerJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, innerJoin)
}

// LeftJoin returns every left row once per match, or once with null right
// values when unmatched. Rows are left-major. Invalid keys and output schemas
// return errors. Output schema follows InnerJoin; right-side output fields are
// nullable.
func (f Frame) LeftJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, leftJoin)
}

// RightJoin returns every right row once per match, or once with null left
// values when unmatched. Rows are right-major. Invalid keys and output schemas
// return errors. Output schema remains left then right; left-side output fields
// are nullable. A coalesced key takes its value from the right key for an
// unmatched right row.
func (f Frame) RightJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, rightJoin)
}

// FullJoin scans left rows in order, emitting their matches or one unmatched
// row, then emits unmatched right rows in right-row order. Invalid keys and
// output schemas return errors. Output schema follows InnerJoin; non-key fields
// from both sides are nullable. A coalesced key takes the present key from
// either side.
func (f Frame) FullJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, fullJoin)
}

// SemiJoin keeps left rows having at least one right match. It emits no right
// columns, does not multiply left rows, and returns the left schema unchanged.
func (f Frame) SemiJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, semiJoin)
}

// AntiJoin keeps left rows having no right match and returns the left schema
// unchanged.
func (f Frame) AntiJoin[K any](right Frame, key JoinKey[K]) (Frame, error) {
	return joinFrames(f, right, key, antiJoin)
}

// CrossJoin returns the Cartesian product. Name collisions return
// ErrColumnConflict.
func (f Frame) CrossJoin(right Frame) (Frame, error) {
	if err := validateColumnNames(f, right, -1, -1, ""); err != nil {
		return Frame{}, err
	}
	if right.Len() != 0 && f.Len() > math.MaxInt/right.Len() {
		return Frame{}, fmt.Errorf("%w: cross join row count overflows int", ErrRowCount)
	}
	total := f.Len() * right.Len()
	leftRows := make([]int, 0, total)
	rightRows := make([]int, 0, total)
	for leftRow := 0; leftRow < f.Len(); leftRow++ {
		for rightRow := 0; rightRow < right.Len(); rightRow++ {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, rightRow)
		}
	}
	columns := make([]column, 0, f.Width()+right.Width())
	for _, column := range f.columns {
		column.length = total
		column.values = column.values.take(leftRows)
		columns = append(columns, column)
	}
	for _, column := range right.columns {
		column.length = total
		column.values = column.values.take(rightRows)
		columns = append(columns, column)
	}
	return Frame{columns: columns, rowCount: total}, nil
}

type joinKind uint8

const (
	innerJoin joinKind = iota
	leftJoin
	rightJoin
	fullJoin
	semiJoin
	antiJoin
)

type joinProjection struct {
	leftKeyIndex  int
	rightKeyIndex int
	outputName    string
}

// resolvedJoinPlan is the normalized runtime form of a JoinKey. Positional
// joins retain the default -1 projection indexes; stored joins resolve their
// key series and output projection once before matching begins.
type resolvedJoinPlan[K any] struct {
	left       series.Series[K]
	right      series.Series[K]
	newLookup  func(int) joinLookup[K]
	projection joinProjection
}

type joinMatch struct {
	first int
	last  int
}

type joinLookup[K any] interface {
	Get(K) (joinMatch, bool)
	Set(K, joinMatch)
}

type comparableJoinLookup[K comparable] map[K]joinMatch

func (m comparableJoinLookup[K]) Get(key K) (joinMatch, bool) {
	match, found := m[key]
	return match, found
}

func (m comparableJoinLookup[K]) Set(key K, match joinMatch) {
	m[key] = match
}

type joinIndex[K any] struct {
	lookup joinLookup[K]
	next   []int
}

func newJoinIndex[K any](right series.Series[K], lookup joinLookup[K]) joinIndex[K] {
	index := joinIndex[K]{lookup: lookup, next: make([]int, right.Len())}
	for i := range index.next {
		index.next[i] = -1
	}
	for row := 0; row < right.Len(); row++ {
		value, present := right.At(row)
		if !present {
			continue
		}
		matches, found := index.lookup.Get(value)
		if !found {
			index.lookup.Set(value, joinMatch{first: row, last: row})
			continue
		}
		index.next[matches.last] = row
		matches.last = row
		index.lookup.Set(value, matches)
	}
	return index
}

func joinFrames[K any](left, right Frame, key JoinKey[K], kind joinKind) (Frame, error) {
	plan, err := resolveJoinPlan(left, right, key)
	if err != nil {
		return Frame{}, err
	}
	if kind != semiJoin && kind != antiJoin {
		projection := plan.projection
		if err := validateColumnNames(left, right, projection.leftKeyIndex, projection.rightKeyIndex, projection.outputName); err != nil {
			return Frame{}, err
		}
	}

	switch kind {
	case innerJoin, leftJoin:
		lookup := plan.newLookup(plan.right.Len())
		leftRows, rightRows := pairedMatchingRows(plan.left, plan.right, lookup, kind)
		return buildJoinFrame(left, right, plan, leftRows, rightRows, kind), nil
	case rightJoin:
		lookup := plan.newLookup(plan.left.Len())
		rightRows, leftRows := pairedMatchingRows(plan.right, plan.left, lookup, kind)
		return buildJoinFrame(left, right, plan, leftRows, rightRows, kind), nil
	case fullJoin:
		lookup := plan.newLookup(plan.right.Len())
		leftRows, rightRows := fullMatchingRows(plan.left, plan.right, lookup)
		return buildJoinFrame(left, right, plan, leftRows, rightRows, kind), nil
	case semiJoin, antiJoin:
		lookup := plan.newLookup(plan.right.Len())
		return left.Take(existenceRows(plan.left, plan.right, lookup, kind)), nil
	default:
		panic("dataframe: invalid join kind")
	}
}

func resolveJoinPlan[K any](left, right Frame, key JoinKey[K]) (resolvedJoinPlan[K], error) {
	plan := resolvedJoinPlan[K]{
		left:      key.left,
		right:     key.right,
		newLookup: key.newLookup,
		projection: joinProjection{
			leftKeyIndex:  -1,
			rightKeyIndex: -1,
		},
	}
	if key.newLookup == nil {
		return resolvedJoinPlan[K]{}, fmt.Errorf("%w: join key has a nil hasher", ErrUnsupported)
	}
	if !key.usingStored {
		if key.left.Len() != left.Len() || key.right.Len() != right.Len() {
			return resolvedJoinPlan[K]{}, fmt.Errorf("%w: positional join keys have lengths %d and %d, want %d and %d", ErrRowCount, key.left.Len(), key.right.Len(), left.Len(), right.Len())
		}
		return plan, nil
	}
	if key.leftName == "" || key.rightName == "" || key.outputName == "" {
		return resolvedJoinPlan[K]{}, fmt.Errorf("%w: join key names must not be empty", ErrInvalidName)
	}
	leftKey, err := left.Column[K](key.leftName)
	if err != nil {
		return resolvedJoinPlan[K]{}, err
	}
	rightKey, err := right.Column[K](key.rightName)
	if err != nil {
		return resolvedJoinPlan[K]{}, err
	}
	plan.left = leftKey
	plan.right = rightKey
	plan.projection = joinProjection{
		leftKeyIndex:  left.columnIndex(key.leftName),
		rightKeyIndex: right.columnIndex(key.rightName),
		outputName:    key.outputName,
	}
	return plan, nil
}

func validateColumnNames(left, right Frame, leftKeyIndex, rightKeyIndex int, outputName string) error {
	names := make(map[string]struct{}, left.Width()+right.Width())
	for i, column := range left.columns {
		name := column.name
		if i == leftKeyIndex {
			name = outputName
		}
		if _, exists := names[name]; exists {
			return fmt.Errorf("%w: %q", ErrColumnConflict, name)
		}
		names[name] = struct{}{}
	}
	for i, column := range right.columns {
		if i == rightKeyIndex {
			continue
		}
		name := column.name
		if _, exists := names[name]; exists {
			return fmt.Errorf("%w: %q", ErrColumnConflict, name)
		}
		names[name] = struct{}{}
	}
	return nil
}

func pairedMatchingRows[K any](probe, indexed series.Series[K], lookup joinLookup[K], kind joinKind) ([]int, []int) {
	if kind != innerJoin && kind != leftJoin && kind != rightJoin {
		panic("dataframe: invalid paired join kind")
	}
	index := newJoinIndex(indexed, lookup)
	probeRows := make([]int, 0, probe.Len())
	indexedRows := make([]int, 0, probe.Len())
	for probeRow := 0; probeRow < probe.Len(); probeRow++ {
		value, present := probe.At(probeRow)
		if !present {
			if kind != innerJoin {
				probeRows = append(probeRows, probeRow)
				indexedRows = append(indexedRows, -1)
			}
			continue
		}
		matches, found := index.lookup.Get(value)
		if !found {
			if kind != innerJoin {
				probeRows = append(probeRows, probeRow)
				indexedRows = append(indexedRows, -1)
			}
			continue
		}
		for indexedRow := matches.first; indexedRow >= 0; indexedRow = index.next[indexedRow] {
			probeRows = append(probeRows, probeRow)
			indexedRows = append(indexedRows, indexedRow)
		}
	}
	return probeRows, indexedRows
}

func fullMatchingRows[K any](left, right series.Series[K], lookup joinLookup[K]) ([]int, []int) {
	index := newJoinIndex(right, lookup)
	leftRows := make([]int, 0, left.Len()+len(index.next))
	rightRows := make([]int, 0, left.Len()+len(index.next))
	matchedRight := make([]bool, len(index.next))
	for leftRow := 0; leftRow < left.Len(); leftRow++ {
		value, present := left.At(leftRow)
		if !present {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, -1)
			continue
		}
		matches, found := index.lookup.Get(value)
		if !found {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, -1)
			continue
		}
		for rightRow := matches.first; rightRow >= 0; rightRow = index.next[rightRow] {
			leftRows = append(leftRows, leftRow)
			rightRows = append(rightRows, rightRow)
			matchedRight[rightRow] = true
		}
	}
	for rightRow, matched := range matchedRight {
		if !matched {
			leftRows = append(leftRows, -1)
			rightRows = append(rightRows, rightRow)
		}
	}
	return leftRows, rightRows
}

func existenceRows[K any](left, right series.Series[K], lookup joinLookup[K], kind joinKind) []int {
	if kind != semiJoin && kind != antiJoin {
		panic("dataframe: invalid existence join kind")
	}
	for row := 0; row < right.Len(); row++ {
		value, present := right.At(row)
		if present {
			lookup.Set(value, joinMatch{})
		}
	}
	rows := make([]int, 0, left.Len())
	for row := 0; row < left.Len(); row++ {
		value, present := left.At(row)
		if !present {
			if kind == antiJoin {
				rows = append(rows, row)
			}
			continue
		}
		_, found := lookup.Get(value)
		if (kind == semiJoin && found) || (kind == antiJoin && !found) {
			rows = append(rows, row)
		}
	}
	return rows
}

func buildJoinFrame[K any](left, right Frame, plan resolvedJoinPlan[K], leftRows, rightRows []int, kind joinKind) Frame {
	var nullableLeftRows series.Series[int]
	var nullableRightRows series.Series[int]
	switch kind {
	case leftJoin:
		nullableRightRows = joinRowIndexes(rightRows)
	case rightJoin:
		nullableLeftRows = joinRowIndexes(leftRows)
	case fullJoin:
		nullableLeftRows = joinRowIndexes(leftRows)
		nullableRightRows = joinRowIndexes(rightRows)
	}

	columns := make([]column, 0, left.Width()+right.Width())
	for i, column := range left.columns {
		if i == plan.projection.leftKeyIndex {
			joined := joinedKey(plan, leftRows, rightRows, nullableLeftRows, nullableRightRows, kind)
			columns = append(columns, typedColumn(plan.projection.outputName, joined))
			continue
		}
		if kind == rightJoin || kind == fullJoin {
			column.nullable = true
			column.values = column.values.takeNullable(nullableLeftRows)
		} else {
			column.values = column.values.take(leftRows)
		}
		column.length = len(leftRows)
		columns = append(columns, column)
	}
	for i, column := range right.columns {
		if i == plan.projection.rightKeyIndex {
			continue
		}
		if kind == leftJoin || kind == fullJoin {
			column.nullable = true
			column.values = column.values.takeNullable(nullableRightRows)
		} else {
			column.values = column.values.take(rightRows)
		}
		column.length = len(leftRows)
		columns = append(columns, column)
	}
	return Frame{columns: columns, rowCount: len(leftRows)}
}

func joinedKey[K any](plan resolvedJoinPlan[K], leftRows, rightRows []int, nullableLeftRows, nullableRightRows series.Series[int], kind joinKind) series.Series[K] {
	switch kind {
	case innerJoin:
		left := plan.left.Take(leftRows)
		if !plan.left.Nullable() && !plan.right.Nullable() {
			return left
		}
		return coalesceJoinKeys(left, plan.right.Take(rightRows))
	case leftJoin:
		return plan.left.Take(leftRows)
	case rightJoin:
		return plan.right.Take(rightRows)
	case fullJoin:
		result := coalesceJoinKeys(plan.left.TakeNullable(nullableLeftRows), plan.right.TakeNullable(nullableRightRows))
		if !plan.left.Nullable() && !plan.right.Nullable() {
			return result.DropNull()
		}
		return result
	default:
		panic("dataframe: invalid join kind")
	}
}

func coalesceJoinKeys[K any](left, right series.Series[K]) series.Series[K] {
	return left.Map2Cells(right, func(left, right series.Optional[K]) series.Optional[K] {
		if left.Valid {
			return left
		}
		return right
	})
}

func joinRowIndexes(rows []int) series.Series[int] {
	values := make([]int, len(rows))
	validity := make([]bool, len(rows))
	for i, row := range rows {
		if row >= 0 {
			values[i] = row
			validity[i] = true
		}
	}
	result, err := series.NewNullable(values, validity)
	if err != nil {
		panic(err)
	}
	return result
}
