package sql

import (
	"bytes"
	"context"
	stdsql "database/sql"
	"database/sql/driver"
	"errors"
	"io"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/joeychilson/dataframe"
	"github.com/joeychilson/dataframe/series"
)

func TestReadUsesMetadataForEntirelyNullColumns(t *testing.T) {
	type nullableCount struct {
		Valid bool
		Value uint16
	}
	type unrelated struct {
		Value int
		Set   bool
	}
	db, _ := openTestDB(map[string]queryResult{
		"nulls": {
			names: []string{"count", "label", "created"},
			scanTypes: []reflect.Type{
				reflect.TypeFor[stdsql.NullInt64](),
				reflect.TypeFor[stdsql.NullString](),
				reflect.TypeFor[stdsql.NullTime](),
			},
			nullable: []bool{true, true, true},
			rows:     [][]driver.Value{{nil, nil, nil}},
		},
		"empty": {
			names: []string{"active", "amount", "generic", "custom", "unrelated"},
			scanTypes: []reflect.Type{
				reflect.TypeFor[bool](),
				reflect.TypeFor[float32](),
				reflect.TypeFor[stdsql.Null[int64]](),
				reflect.TypeFor[*nullableCount](),
				reflect.TypeFor[unrelated](),
			},
			nullable: []bool{false, true, true, true, true},
		},
	})
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})

	frame, err := Query(context.Background(), db, "nulls")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := frame.Schema(), []dataframe.Field{
		{Name: "count", Type: reflect.TypeFor[int64](), Nullable: true},
		{Name: "label", Type: reflect.TypeFor[string](), Nullable: true},
		{Name: "created", Type: reflect.TypeFor[time.Time](), Nullable: true},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("all-null schema = %#v, want %#v", got, want)
	}

	frame, err = Query(context.Background(), db, "empty")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := frame.Schema(), []dataframe.Field{
		{Name: "active", Type: reflect.TypeFor[bool]()},
		{Name: "amount", Type: reflect.TypeFor[float64](), Nullable: true},
		{Name: "generic", Type: reflect.TypeFor[int64](), Nullable: true},
		{Name: "custom", Type: reflect.TypeFor[int64](), Nullable: true},
		{Name: "unrelated", Type: reflect.TypeFor[string](), Nullable: true},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("empty schema = %#v, want %#v", got, want)
	}
}

func TestReadPadsLeadingNullsWhenTypeAppearsLater(t *testing.T) {
	db, _ := openTestDB(map[string]queryResult{
		"values": {
			names:     []string{"active"},
			scanTypes: []reflect.Type{reflect.TypeFor[bool]()},
			nullable:  []bool{true},
			rows:      [][]driver.Value{{nil}, {true}},
		},
	})
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})

	frame, err := Query(context.Background(), db, "values")
	if err != nil {
		t.Fatal(err)
	}
	values, err := frame.Column[bool]("active")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := values.Optionals(), []series.Optional[bool]{series.None[bool](), series.Some(true)}; !reflect.DeepEqual(got, want) {
		t.Fatalf("values = %#v, want %#v", got, want)
	}
}

func TestReadAcrossColumnBlocks(t *testing.T) {
	blockRows := int(uintptr(sqlColumnTargetBlockBytes) / reflect.TypeFor[int64]().Size())
	rows := make([][]driver.Value, blockRows+1)
	for i := range blockRows {
		rows[i] = []driver.Value{int64(i), nil}
	}
	rows[blockRows] = []driver.Value{int64(blockRows), int64(7)}
	db, _ := openTestDB(map[string]queryResult{
		"values": {
			names:     []string{"id", "score"},
			scanTypes: []reflect.Type{reflect.TypeFor[int64](), reflect.TypeFor[int64]()},
			nullable:  []bool{false, true},
			rows:      rows,
		},
	})
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})

	frame, err := Query(context.Background(), db, "values")
	if err != nil {
		t.Fatal(err)
	}
	ids, err := frame.Column[int64]("id")
	if err != nil {
		t.Fatal(err)
	}
	scores, err := frame.Column[int64]("score")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := frame.Len(), blockRows+1; got != want {
		t.Fatalf("row count = %d, want %d", got, want)
	}
	for _, row := range []int{blockRows - 1, blockRows} {
		if value, ok := ids.At(row); !ok || value != int64(row) {
			t.Fatalf("id at row %d = %d, %t", row, value, ok)
		}
	}
	if value, ok := scores.At(blockRows - 1); ok || value != 0 {
		t.Fatalf("score before boundary = %d, %t", value, ok)
	}
	if value, ok := scores.At(blockRows); !ok || value != 7 {
		t.Fatalf("score after boundary = %d, %t", value, ok)
	}
}

func TestReadRejectsInvalidAndChangingColumns(t *testing.T) {
	closed := false
	db, _ := openTestDB(map[string]queryResult{
		"mixed": {
			names:     []string{"value"},
			scanTypes: []reflect.Type{reflect.TypeFor[any]()},
			rows:      [][]driver.Value{{int64(1)}, {"two"}},
			closed:    &closed,
		},
		"duplicate": {
			names:     []string{"value", "value"},
			scanTypes: []reflect.Type{reflect.TypeFor[string](), reflect.TypeFor[string]()},
		},
	})
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})

	if _, err := Query(context.Background(), db, "mixed"); !errors.Is(err, dataframe.ErrColumnType) {
		t.Fatalf("mixed type error = %v", err)
	}
	if !closed {
		t.Fatal("rows were not closed after conversion error")
	}
	if _, err := Query(context.Background(), db, "duplicate"); !errors.Is(err, dataframe.ErrColumnConflict) {
		t.Fatalf("duplicate name error = %v", err)
	}
}

func TestWriterExecutesFrameRowsInSchemaOrder(t *testing.T) {
	db, state := openTestDB(nil)
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})
	statement, err := db.PrepareContext(context.Background(), "insert")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := statement.Close(); closeErr != nil {
			t.Errorf("close statement: %v", closeErr)
		}
	})

	names := series.FromOptionals([]series.Optional[string]{series.Some("A"), series.None[string]()})
	frame, err := dataframe.New(
		dataframe.Column("id", []int{1, 2}),
		dataframe.ColumnFromSeries("name", names),
		dataframe.Column("code", []valuerCode{7, 8}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if writeErr := Write(context.Background(), statement, frame); writeErr != nil {
		t.Fatal(writeErr)
	}
	want := [][]driver.Value{{int64(1), "A", "C7"}, {int64(2), nil, "C8"}}
	if !reflect.DeepEqual(state.executed, want) {
		t.Fatalf("executed = %#v, want %#v", state.executed, want)
	}
}

func TestWriterWritesRecordsAndReportsRow(t *testing.T) {
	type input struct {
		ID   int          `df:"id"`
		Name *string      `df:"name"`
		Code valuerCode   `df:"code"`
		Note valuerString `df:"note"`
	}
	name := "A"
	db, state := openTestDB(nil)
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})
	statement, err := db.PrepareContext(context.Background(), "insert")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := statement.Close(); closeErr != nil {
			t.Errorf("close statement: %v", closeErr)
		}
	})

	writer := NewWriter(statement)
	if writeErr := writer.WriteRecords(context.Background(), []input{
		{ID: 1, Name: &name, Code: 7, Note: "first"},
		{ID: 2, Code: 8, Note: "second"},
	}); writeErr != nil {
		t.Fatal(writeErr)
	}
	want := [][]driver.Value{{int64(1), "A", "C7", "V:first"}, {int64(2), nil, "C8", "V:second"}}
	if !reflect.DeepEqual(state.executed, want) {
		t.Fatalf("executed = %#v, want %#v", state.executed, want)
	}

	state.failAt = len(state.executed) + 1
	err = writer.WriteRecords(context.Background(), []input{{ID: 3}, {ID: 4}})
	if err == nil || !strings.Contains(err.Error(), "row 1") {
		t.Fatalf("execution error = %v", err)
	}
}

func TestWriterWriteRecordsCopiesAndUnwrapsValuers(t *testing.T) {
	type input struct {
		Mutable  incrementingValuerCode `df:"mutable"`
		Dynamic  any                    `df:"dynamic"`
		Optional series.Optional[any]   `df:"optional"`
	}
	records := []input{{
		Mutable:  7,
		Dynamic:  valuerCode(7),
		Optional: series.Some[any](valuerCode(8)),
	}}
	db, state := openTestDB(nil)
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})
	statement, err := db.PrepareContext(context.Background(), "insert")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := statement.Close(); closeErr != nil {
			t.Errorf("close statement: %v", closeErr)
		}
	})

	if writeErr := NewWriter(statement).WriteRecords(context.Background(), records); writeErr != nil {
		t.Fatal(writeErr)
	}
	want := [][]driver.Value{{"C8", "C7", "C8"}}
	if !reflect.DeepEqual(state.executed, want) {
		t.Fatalf("executed = %#v, want %#v", state.executed, want)
	}
	if records[0].Mutable != 7 {
		t.Fatalf("record value mutated to %d", records[0].Mutable)
	}
}

func TestQueryReadsNativeValuesAndNulls(t *testing.T) {
	now := time.Date(2026, time.August, 22, 12, 30, 0, 0, time.UTC)
	payload := []byte("one")
	closed := false
	db, _ := openTestDB(map[string]queryResult{
		"report": {
			names:     []string{"id", "name", "score", "created", "payload"},
			scanTypes: []reflect.Type{reflect.TypeFor[int64](), reflect.TypeFor[string](), reflect.TypeFor[float64](), reflect.TypeFor[time.Time](), reflect.TypeFor[[]byte]()},
			nullable:  []bool{false, true, true, false, true},
			rows: [][]driver.Value{
				{int64(1), "A", 1.5, now, payload},
				{int64(2), nil, nil, now.Add(time.Hour), nil},
			},
			closed: &closed,
		},
	})
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})

	frame, err := Query(context.Background(), db, "report")
	if err != nil {
		t.Fatal(err)
	}
	if !closed {
		t.Fatal("query rows were not closed")
	}
	wantSchema := []dataframe.Field{
		{Name: "id", Type: reflect.TypeFor[int64]()},
		{Name: "name", Type: reflect.TypeFor[string](), Nullable: true},
		{Name: "score", Type: reflect.TypeFor[float64](), Nullable: true},
		{Name: "created", Type: reflect.TypeFor[time.Time]()},
		{Name: "payload", Type: reflect.TypeFor[[]byte](), Nullable: true},
	}
	if got := frame.Schema(); !reflect.DeepEqual(got, wantSchema) {
		t.Fatalf("schema = %#v, want %#v", got, wantSchema)
	}
	ids, _ := frame.Column[int64]("id")
	names, _ := frame.Column[string]("name")
	scores, _ := frame.Column[float64]("score")
	payloads, _ := frame.Column[[]byte]("payload")
	if !slices.Equal(ids.Values(), []int64{1, 2}) || !reflect.DeepEqual(names.Optionals(), []series.Optional[string]{series.Some("A"), series.None[string]()}) || !reflect.DeepEqual(scores.Optionals(), []series.Optional[float64]{series.Some(1.5), series.None[float64]()}) {
		t.Fatalf("values = %v, %#v, %#v", ids.Values(), names.Optionals(), scores.Optionals())
	}
	if value, ok := payloads.At(0); !ok || string(value) != "one" {
		t.Fatalf("payload = %q, %v", value, ok)
	}
	// A driver may reuse byte storage after Scan returns. The frame must own a
	// stable copy of every present byte slice.
	payload[0] = 'x'
	if value, _ := payloads.At(0); string(value) != "one" {
		t.Fatalf("payload changed with driver storage: %q", value)
	}
}

func TestQueryRecordsUsesDatabaseSQLScanning(t *testing.T) {
	type result struct {
		ID      int                      `df:"id"`
		Name    *string                  `df:"name"`
		Score   series.Optional[float64] `df:"score"`
		Code    scanCode                 `df:"code"`
		Comment stdsql.NullString        `df:"comment"`
	}
	db, _ := openTestDB(map[string]queryResult{
		"records": {
			names:     []string{"id", "name", "score", "code", "comment", "ignored"},
			scanTypes: []reflect.Type{reflect.TypeFor[int64](), reflect.TypeFor[string](), reflect.TypeFor[float64](), reflect.TypeFor[string](), reflect.TypeFor[string](), reflect.TypeFor[string]()},
			rows: [][]driver.Value{
				{int64(1), "A", 1.5, "C7", "good", "x"},
				{int64(2), nil, nil, "C8", nil, "y"},
			},
		},
		"bad-null": {
			names:     []string{"id", "name", "score", "code", "comment"},
			scanTypes: []reflect.Type{reflect.TypeFor[int64](), reflect.TypeFor[string](), reflect.TypeFor[float64](), reflect.TypeFor[string](), reflect.TypeFor[string]()},
			rows:      [][]driver.Value{{nil, "A", 1.5, "C7", nil}},
		},
		"missing-fields": {
			names:     []string{"id", "name", "ignored"},
			scanTypes: []reflect.Type{reflect.TypeFor[int64](), reflect.TypeFor[string](), reflect.TypeFor[string]()},
		},
	})
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})

	records, err := QueryRecords[result](context.Background(), db, "records")
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 || records[0].ID != 1 || records[0].Name == nil || *records[0].Name != "A" || records[1].Name != nil || !records[0].Score.Valid || records[1].Score.Valid || records[0].Code != 7 || records[1].Code != 8 || records[0].Comment.String != "good" || !records[0].Comment.Valid || records[1].Comment.Valid {
		t.Fatalf("records = %#v", records)
	}
	if _, queryErr := QueryRecords[result](context.Background(), db, "bad-null"); !errors.Is(queryErr, dataframe.ErrInvalidRecord) {
		t.Fatalf("non-null field error = %v", queryErr)
	}
	if _, queryErr := QueryRecords[result](context.Background(), db, "missing-fields"); !errors.Is(queryErr, dataframe.ErrColumnNotFound) || !strings.Contains(queryErr.Error(), `"score"`) {
		t.Fatalf("missing field error = %v", queryErr)
	}
}

func TestNilAndOneShotErrors_AreReported(t *testing.T) {
	if _, err := Read(nil); err == nil {
		t.Fatal("nil rows did not fail")
	}
	if _, err := Query(context.Background(), (*stdsql.DB)(nil), "query"); err == nil {
		t.Fatal("nil queryer did not fail")
	}
	if err := Write(context.Background(), (*stdsql.Stmt)(nil), dataframe.Frame{}); err == nil {
		t.Fatal("nil statement did not fail")
	}

	db, _ := openTestDB(map[string]queryResult{
		"once": {names: []string{"value"}, scanTypes: []reflect.Type{reflect.TypeFor[string]()}},
	})
	t.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			t.Errorf("close database: %v", closeErr)
		}
	})
	rows, err := db.QueryContext(context.Background(), "once")
	if err != nil {
		t.Fatal(err)
	}
	reader := NewReader(rows)
	if _, readErr := reader.Read(); readErr != nil {
		t.Fatal(readErr)
	}
	if _, readErr := reader.Read(); readErr == nil {
		t.Fatal("second read did not fail")
	}
}

type scanCode int

func (c *scanCode) Scan(source any) error {
	text, ok := source.(string)
	if !ok || len(text) < 2 || text[0] != 'C' {
		return errors.New("invalid code")
	}
	*c = scanCode(text[1] - '0')
	return nil
}

type valuerCode int

func (c *valuerCode) Value() (driver.Value, error) {
	return "C" + string(rune('0'+*c)), nil
}

type incrementingValuerCode int

func (c *incrementingValuerCode) Value() (driver.Value, error) {
	*c++
	return "C" + string(rune('0'+*c)), nil
}

type valuerString string

func (s *valuerString) Value() (driver.Value, error) {
	return "V:" + string(*s), nil
}

type queryResult struct {
	names     []string
	scanTypes []reflect.Type
	nullable  []bool
	rows      [][]driver.Value
	closed    *bool
}

type testState struct {
	queries  map[string]queryResult
	executed [][]driver.Value
	failAt   int
}

func openTestDB(queries map[string]queryResult) (*stdsql.DB, *testState) {
	state := &testState{queries: queries}
	return stdsql.OpenDB(testConnector{state: state}), state
}

type testConnector struct {
	state *testState
}

func (c testConnector) Connect(context.Context) (driver.Conn, error) {
	return &testConn{state: c.state}, nil
}

func (c testConnector) Driver() driver.Driver {
	return testDriver{}
}

type testDriver struct{}

func (testDriver) Open(string) (driver.Conn, error) {
	return nil, errors.New("use connector")
}

type testConn struct {
	state *testState
}

func (c *testConn) Prepare(query string) (driver.Stmt, error) {
	return &testStatement{state: c.state}, nil
}

func (c *testConn) Close() error { return nil }

func (c *testConn) Begin() (driver.Tx, error) { return nil, errors.New("transactions unsupported") }

func (c *testConn) QueryContext(_ context.Context, query string, _ []driver.NamedValue) (driver.Rows, error) {
	result, ok := c.state.queries[query]
	if !ok {
		return nil, errors.New("unknown query")
	}
	return &testRows{queryResult: result}, nil
}

type testStatement struct {
	state *testState
}

func (s *testStatement) Close() error { return nil }

func (s *testStatement) NumInput() int { return -1 }

func (s *testStatement) Exec(arguments []driver.Value) (driver.Result, error) {
	return s.execute(arguments)
}

func (s *testStatement) ExecContext(_ context.Context, arguments []driver.NamedValue) (driver.Result, error) {
	values := make([]driver.Value, len(arguments))
	for i := range arguments {
		values[i] = arguments[i].Value
	}
	return s.execute(values)
}

func (s *testStatement) execute(arguments []driver.Value) (driver.Result, error) {
	call := len(s.state.executed)
	if s.state.failAt > 0 && call == s.state.failAt {
		return nil, errors.New("forced execution failure")
	}
	s.state.executed = append(s.state.executed, slices.Clone(arguments))
	return driver.RowsAffected(1), nil
}

func (s *testStatement) Query([]driver.Value) (driver.Rows, error) {
	return nil, errors.New("queries unsupported")
}

type testRows struct {
	queryResult
	position int
}

func (r *testRows) Columns() []string { return slices.Clone(r.names) }

func (r *testRows) Close() error {
	if r.closed != nil {
		*r.closed = true
	}
	return nil
}

func (r *testRows) Next(destination []driver.Value) error {
	if r.position >= len(r.rows) {
		return io.EOF
	}
	copy(destination, r.rows[r.position])
	r.position++
	return nil
}

func (r *testRows) ColumnTypeScanType(index int) reflect.Type {
	return r.scanTypes[index]
}

func (r *testRows) ColumnTypeNullable(index int) (bool, bool) {
	if r.nullable == nil {
		return false, false
	}
	return r.nullable[index], true
}

func BenchmarkReadFrame(b *testing.B) {
	payload := []byte(strings.Repeat("x", 128))
	rows := make([][]driver.Value, 10_000)
	for i := range rows {
		rows[i] = []driver.Value{int64(i), "value", float64(i), payload}
	}
	db, _ := openTestDB(map[string]queryResult{
		"frame": {
			names:     []string{"id", "name", "score", "payload"},
			scanTypes: []reflect.Type{reflect.TypeFor[int64](), reflect.TypeFor[string](), reflect.TypeFor[float64](), reflect.TypeFor[[]byte]()},
			rows:      rows,
		},
	})
	b.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			b.Errorf("close database: %v", closeErr)
		}
	})
	ctx := context.Background()
	b.ReportAllocs()
	var result dataframe.Frame
	for b.Loop() {
		var err error
		result, err = Query(ctx, db, "frame")
		if err != nil {
			b.Fatal(err)
		}
	}
	runtime.KeepAlive(result)
}

func BenchmarkScannedColumnStorage(b *testing.B) {
	const size = 10_000
	inputs := []struct {
		name   string
		typeOf reflect.Type
		values []any
	}{
		{name: "Bool", typeOf: reflect.TypeFor[bool](), values: make([]any, size)},
		{name: "Int64", typeOf: reflect.TypeFor[int64](), values: make([]any, size)},
		{name: "Float64", typeOf: reflect.TypeFor[float64](), values: make([]any, size)},
		{name: "String", typeOf: reflect.TypeFor[string](), values: make([]any, size)},
		{name: "Bytes", typeOf: reflect.TypeFor[[]byte](), values: make([]any, size)},
		{name: "Time", typeOf: reflect.TypeFor[time.Time](), values: make([]any, size)},
		{name: "LeadingNulls", typeOf: reflect.TypeFor[int64](), values: make([]any, size)},
		{name: "SparseNulls", typeOf: reflect.TypeFor[int64](), values: make([]any, size)},
		{name: "AllNull", typeOf: reflect.TypeFor[string](), values: make([]any, size)},
	}
	for _, input := range inputs {
		for i := range input.values {
			switch input.name {
			case "Bool":
				input.values[i] = i%2 == 0
			case "Int64":
				input.values[i] = int64(i)
			case "Float64":
				input.values[i] = float64(i) / 3
			case "String":
				input.values[i] = "value"
			case "Bytes":
				input.values[i] = []byte("value")
			case "Time":
				input.values[i] = time.Unix(int64(i), 0)
			case "LeadingNulls":
				if i >= 100 {
					input.values[i] = int64(i)
				}
			case "SparseNulls":
				if i%4 != 0 {
					input.values[i] = int64(i)
				}
			}
		}
	}
	consume := func(value any) int64 {
		switch typed := value.(type) {
		case bool:
			if typed {
				return 1
			}
		case int64:
			return typed
		case float64:
			return int64(typed)
		case string:
			return int64(len(typed))
		case []byte:
			return int64(len(typed))
		case time.Time:
			return typed.Unix()
		}
		return 0
	}
	for _, input := range inputs {
		b.Run(input.name, func(b *testing.B) {
			b.Run("Optimized", func(b *testing.B) {
				b.ReportAllocs()
				var total int64
				for b.Loop() {
					column := scannedColumn{name: "value"}
					for _, value := range input.values {
						if scanErr := column.Scan(value); scanErr != nil {
							b.Fatal(scanErr)
						}
					}
					if column.typeOf == nil {
						column.typeOf = input.typeOf
						column.appendZeroes(column.length)
					}
					for row := range column.length {
						switch input.name {
						case "Bool":
							total += consume(column.bools.at(row))
						case "Int64", "LeadingNulls", "SparseNulls":
							total += consume(column.ints.at(row))
						case "Float64":
							total += consume(column.floats.at(row))
						case "String", "AllNull":
							total += consume(column.strings.at(row))
						case "Bytes":
							total += consume(column.bytes.at(row))
						case "Time":
							total += consume(column.times.at(row))
						}
					}
				}
				runtime.KeepAlive(total)
			})
			b.Run("Reference", func(b *testing.B) {
				b.ReportAllocs()
				var total int64
				for b.Loop() {
					var values []any
					var typeOf reflect.Type
					for _, value := range input.values {
						if value != nil {
							valueType := reflect.TypeOf(value)
							if typeOf == nil {
								typeOf = valueType
							} else if typeOf != valueType {
								b.Fatalf("column changed from %v to %v", typeOf, valueType)
							}
							if bytesValue, ok := value.([]byte); ok {
								value = bytes.Clone(bytesValue)
							}
						}
						values = append(values, value)
					}
					for _, value := range values {
						total += consume(value)
					}
				}
				runtime.KeepAlive(total)
			})
		})
	}
}

func BenchmarkReadRecords(b *testing.B) {
	type result struct {
		ID    int64   `df:"id"`
		Name  string  `df:"name"`
		Score float64 `df:"score"`
	}
	payload := []byte(strings.Repeat("x", 128))
	rows := make([][]driver.Value, 10_000)
	for i := range rows {
		rows[i] = []driver.Value{int64(i), "value", float64(i), payload, payload}
	}
	db, _ := openTestDB(map[string]queryResult{
		"projected": {
			names:     []string{"id", "name", "score"},
			scanTypes: []reflect.Type{reflect.TypeFor[int64](), reflect.TypeFor[string](), reflect.TypeFor[float64]()},
			rows:      rows,
		},
		"ignored-columns": {
			names:     []string{"id", "name", "score", "ignored_a", "ignored_b"},
			scanTypes: []reflect.Type{reflect.TypeFor[int64](), reflect.TypeFor[string](), reflect.TypeFor[float64](), reflect.TypeFor[[]byte](), reflect.TypeFor[[]byte]()},
			rows:      rows,
		},
	})
	b.Cleanup(func() {
		if closeErr := db.Close(); closeErr != nil {
			b.Errorf("close database: %v", closeErr)
		}
	})
	ctx := context.Background()
	for _, query := range []string{"projected", "ignored-columns"} {
		b.Run(query, func(b *testing.B) {
			b.ReportAllocs()
			var records []result
			for b.Loop() {
				scannedRows, queryErr := db.QueryContext(ctx, query)
				if queryErr != nil {
					b.Fatal(queryErr)
				}
				records, queryErr = ReadRecords[result](scannedRows)
				if queryErr != nil {
					b.Fatal(queryErr)
				}
			}
			if len(records) != 10_000 {
				b.Fatalf("record count = %d", len(records))
			}
			runtime.KeepAlive(records)
		})
	}
}
