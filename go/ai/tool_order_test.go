package ai

import (
	"context"
	"slices"
	"testing"

	"github.com/firebase/genkit/go/internal/registry"
)

func TestGeneratePreservesToolOrder(t *testing.T) {
	for _, withMiddleware := range []bool{false, true} {
		name := "declared"
		if withMiddleware {
			name = "declared and middleware"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			reg := registry.New()
			ConfigureFormats(reg)
			DefineGenerateAction(ctx, reg)
			registered := map[string]ToolRef{}
			// Registration order deliberately differs from request order.
			for _, name := range []string{"alpha", "bravo", "charlie", "delta"} {
				registered[name] = defineTool(reg, name, "test tool", func(_ *ToolContext, _ struct{}) (string, error) { return "ok", nil })
			}
			want := []string{"delta", "alpha", "charlie", "bravo"}
			var refs []ToolRef
			for _, name := range want {
				refs = append(refs, registered[name])
			}
			var middleware []Middleware
			if withMiddleware {
				first := NewTool("zulu", "middleware tool", func(_ *ToolContext, _ struct{}) (string, error) { return "ok", nil })
				second := NewTool("echo", "middleware tool", func(_ *ToolContext, _ struct{}) (string, error) { return "ok", nil })
				middleware = append(middleware, MiddlewareFunc(func(context.Context) (*Hooks, error) { return &Hooks{Tools: []Tool{first, second}}, nil }))
				want = append(want, "zulu", "echo")
			}
			calls := 0
			model := defineModel(reg, "test/order-check", &ModelOptions{Supports: &ModelSupports{Tools: true}}, func(_ context.Context, req *ModelRequest, _ ModelStreamCallback) (*ModelResponse, error) {
				calls++
				var got []string
				for _, tool := range req.Tools {
					got = append(got, tool.Name)
				}
				if !slices.Equal(got, want) {
					t.Fatalf("tool order = %v, want %v", got, want)
				}
				return &ModelResponse{Request: req, Message: NewModelTextMessage("ok")}, nil
			})
			for range 100 {
				// Repeated WithTools options append in caller order too.
				_, err := Generate(ctx, reg, WithModel(model), WithTools(refs[:2]...), WithTools(refs[2:]...), WithUse(middleware...), WithPrompt("test"))
				if err != nil {
					t.Fatal(err)
				}
			}
			if calls != 100 {
				t.Fatalf("model calls = %d, want 100", calls)
			}
		})
	}
}
