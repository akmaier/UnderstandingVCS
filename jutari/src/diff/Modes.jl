"""
    Diff

Global HARD vs SOFT execution mode toggle.

HARD: bit-exact emulation against xitari. Integer state, hard opcode dispatch,
indexed memory reads. No gradients. Default and what every conformance test
in `test/conformance/` runs in.

SOFT: relaxed differentiable emulation. Float state, softmax opcode dispatch,
NTM-style soft memory reads/writes, ROM-as-weights. Gradients flow via
Zygote.jl + ChainRulesCore.jl.

The mode is a module-global so the same module code can be reused in both
paths without threading a context through every function. Use `set_mode!` /
`current_mode` to switch, or the `using_mode` do-block.
"""
module Diff

export Mode, current_mode, set_mode!, using_mode

@enum Mode HARD SOFT

const _current = Ref{Mode}(HARD)

current_mode() = _current[]

set_mode!(mode::Mode) = (_current[] = mode; nothing)

"""
    using_mode(f, mode::Mode)

Scoped mode switch. Usage:

    using_mode(SOFT) do
        # ... runs in SOFT mode ...
    end
"""
function using_mode(f, mode::Mode)
    previous = _current[]
    _current[] = mode
    try
        return f()
    finally
        _current[] = previous
    end
end

end # module
