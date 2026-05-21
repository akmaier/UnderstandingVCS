using Test
using JuTari
using JuTari.CPU: step  # qualified to avoid Base.step collision; see JuTari.jl

function _make_memory(image)
    mem = zeros(UInt8, 1 << 16)
    for (addr, value) in image
        mem[(Int(addr) & 0xFFFF) + 1] = UInt8(Int(value) & 0xFF)
    end
    return mem
end

function _state(; PC=0x0000, A=0x00, X=0x00, Y=0x00, SP=0xFD, P=0x34)
    s = initial_cpu_state()
    s.PC = UInt16(PC); s.A = UInt8(A); s.X = UInt8(X); s.Y = UInt8(Y)
    s.SP = UInt8(SP); s.P = UInt8(P)
    s.cycles = UInt64(0)
    return s
end

@testset "JuTari P0 scaffolding" begin

    @testset "package version" begin
        @test JuTari.VERSION_STRING == "0.0.1"
    end

    @testset "initial CPU state matches RESET convention" begin
        s = initial_cpu_state()
        @test s.SP == 0xFD
        @test (s.P & FLAG_I) != 0
        @test (s.P & FLAG_B) != 0
        @test s.cycles == 0
    end

    @testset "step stub on unimplemented opcode (NOP \$EA)" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xEA))
        step(s, mem)
        @test s.PC == 0x8001
        @test s.cycles == 2
    end

    @testset "mode default is HARD and using_mode restores" begin
        @test current_mode() === HARD
        using_mode(SOFT) do
            @test current_mode() === SOFT
        end
        @test current_mode() === HARD
    end

    @testset "set_mode! persists until reset" begin
        set_mode!(SOFT)
        try
            @test current_mode() === SOFT
        finally
            set_mode!(HARD)
        end
    end

end

@testset "JuTari P1a load/store/transfer" begin

    @testset "LDA #imm sets A and clears Z/N" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xA9, 0x8001 => 0x42))
        step(s, mem)
        @test s.A == 0x42
        @test s.PC == 0x8002
        @test s.cycles == 2
        @test (s.P & FLAG_Z) == 0
        @test (s.P & FLAG_N) == 0
        @test (s.P & FLAG_U) != 0
    end

    @testset "LDA #0 sets Z" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xA9, 0x8001 => 0x00))
        step(s, mem)
        @test (s.P & FLAG_Z) != 0
        @test (s.P & FLAG_N) == 0
    end

    @testset "LDA #neg sets N" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xA9, 0x8001 => 0x80))
        step(s, mem)
        @test (s.P & FLAG_N) != 0
        @test (s.P & FLAG_Z) == 0
    end

    @testset "LDA zp,X wraps at \$FF" begin
        s = _state(PC=0x8000, X=0x10)
        mem = _make_memory(Dict(0x8000 => 0xB5, 0x8001 => 0xF5, 0x0005 => 0xAB))
        step(s, mem)
        @test s.A == 0xAB
        @test s.cycles == 4
    end

    @testset "LDA absolute" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xAD, 0x8001 => 0x34, 0x8002 => 0x12, 0x1234 => 0x99))
        step(s, mem)
        @test s.A == 0x99
        @test s.PC == 0x8003
        @test s.cycles == 4
    end

    @testset "LDA absolute,X page cross adds cycle" begin
        s = _state(PC=0x8000, X=0x10)
        mem = _make_memory(Dict(0x8000 => 0xBD, 0x8001 => 0xF5, 0x8002 => 0x12, 0x1305 => 0x55))
        step(s, mem)
        @test s.A == 0x55
        @test s.cycles == 5
    end

    @testset "LDA (zp,X) with zp wrap" begin
        s = _state(PC=0x8000, X=0x04)
        mem = _make_memory(Dict(
            0x8000 => 0xA1, 0x8001 => 0xFE,
            0x0002 => 0x34, 0x0003 => 0x12,
            0x1234 => 0x66,
        ))
        step(s, mem)
        @test s.A == 0x66
        @test s.cycles == 6
    end

    @testset "LDA (zp),Y with zp pointer wrap and page cross" begin
        s = _state(PC=0x8000, Y=0x10)
        mem = _make_memory(Dict(
            0x8000 => 0xB1, 0x8001 => 0xFF,
            0x00FF => 0xF5, 0x0000 => 0x12,
            0x1305 => 0x88,
        ))
        step(s, mem)
        @test s.A == 0x88
        @test s.cycles == 6
    end

    @testset "LDX #neg" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xA2, 0x8001 => 0xFE))
        step(s, mem)
        @test s.X == 0xFE
        @test (s.P & FLAG_N) != 0
    end

    @testset "LDX zp,Y" begin
        s = _state(PC=0x8000, Y=0x02)
        mem = _make_memory(Dict(0x8000 => 0xB6, 0x8001 => 0x10, 0x0012 => 0x07))
        step(s, mem)
        @test s.X == 0x07
        @test s.cycles == 4
    end

    @testset "LDY absolute,X" begin
        s = _state(PC=0x8000, X=0x01)
        mem = _make_memory(Dict(0x8000 => 0xBC, 0x8001 => 0x00, 0x8002 => 0x12, 0x1201 => 0x09))
        step(s, mem)
        @test s.Y == 0x09
        @test s.cycles == 4
    end

    @testset "STA zp writes memory" begin
        s = _state(PC=0x8000, A=0x5A)
        mem = _make_memory(Dict(0x8000 => 0x85, 0x8001 => 0x42))
        step(s, mem)
        @test mem[0x0042 + 1] == 0x5A
        @test s.cycles == 3
    end

    @testset "STA abs,X has NO page-cross penalty" begin
        s = _state(PC=0x8000, A=0xCC, X=0x10)
        before_p = s.P
        mem = _make_memory(Dict(0x8000 => 0x9D, 0x8001 => 0xF5, 0x8002 => 0x12))
        step(s, mem)
        @test mem[0x1305 + 1] == 0xCC
        @test s.cycles == 5
        @test s.P == before_p   # STA never touches flags
    end

    @testset "STX zp,Y" begin
        s = _state(PC=0x8000, X=0x77, Y=0x03)
        mem = _make_memory(Dict(0x8000 => 0x96, 0x8001 => 0x10))
        step(s, mem)
        @test mem[0x0013 + 1] == 0x77
        @test s.cycles == 4
    end

    @testset "STY absolute" begin
        s = _state(PC=0x8000, Y=0x42)
        mem = _make_memory(Dict(0x8000 => 0x8C, 0x8001 => 0x00, 0x8002 => 0x20))
        step(s, mem)
        @test mem[0x2000 + 1] == 0x42
        @test s.PC == 0x8003
    end

    @testset "TAX copies A and sets Z/N" begin
        s = _state(PC=0x8000, A=0x80)
        mem = _make_memory(Dict(0x8000 => 0xAA))
        step(s, mem)
        @test s.X == 0x80
        @test (s.P & FLAG_N) != 0
        @test s.cycles == 2
    end

    @testset "TAY zero sets Z" begin
        s = _state(PC=0x8000, A=0x00)
        mem = _make_memory(Dict(0x8000 => 0xA8))
        step(s, mem)
        @test s.Y == 0x00
        @test (s.P & FLAG_Z) != 0
    end

    @testset "TXA / TYA / TSX chain" begin
        s = _state(PC=0x8000, X=0x01, Y=0x02, SP=0xF0)
        mem = _make_memory(Dict(0x8000 => 0x8A, 0x8001 => 0x98, 0x8002 => 0xBA))
        step(s, mem); @test s.A == 0x01
        step(s, mem); @test s.A == 0x02
        step(s, mem); @test s.X == 0xF0; @test (s.P & FLAG_N) != 0
    end

    @testset "TXS does NOT touch flags" begin
        s = _state(PC=0x8000, X=0x00, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x9A))
        step(s, mem)
        @test s.SP == 0x00
        @test s.P == FLAG_U
    end

end

@testset "JuTari P1b1 bitwise / compare / BIT" begin

    @testset "AND immediate" begin
        s = _state(PC=0x8000, A=0xF0)
        mem = _make_memory(Dict(0x8000 => 0x29, 0x8001 => 0x0F))
        step(s, mem)
        @test s.A == 0x00
        @test (s.P & FLAG_Z) != 0
        @test (s.P & FLAG_N) == 0
        @test s.cycles == 2
    end

    @testset "ORA immediate sets N" begin
        s = _state(PC=0x8000, A=0x01)
        mem = _make_memory(Dict(0x8000 => 0x09, 0x8001 => 0x80))
        step(s, mem)
        @test s.A == 0x81
        @test (s.P & FLAG_N) != 0
        @test (s.P & FLAG_Z) == 0
    end

    @testset "EOR immediate toggles high bit" begin
        s = _state(PC=0x8000, A=0xFF)
        mem = _make_memory(Dict(0x8000 => 0x49, 0x8001 => 0x80))
        step(s, mem)
        @test s.A == 0x7F
        @test (s.P & FLAG_N) == 0
    end

    @testset "AND abs,X page cross adds cycle" begin
        s = _state(PC=0x8000, A=0xFF, X=0x10)
        mem = _make_memory(Dict(0x8000 => 0x3D, 0x8001 => 0xF5, 0x8002 => 0x12, 0x1305 => 0x0F))
        step(s, mem)
        @test s.A == 0x0F
        @test s.cycles == 5
    end

    @testset "ORA (zp),Y no page cross" begin
        s = _state(PC=0x8000, A=0x00, Y=0x01)
        mem = _make_memory(Dict(
            0x8000 => 0x11, 0x8001 => 0x10,
            0x0010 => 0x00, 0x0011 => 0x12,
            0x1201 => 0x42,
        ))
        step(s, mem)
        @test s.A == 0x42
        @test s.cycles == 5
    end

    @testset "CMP immediate equal sets Z and C" begin
        s = _state(PC=0x8000, A=0x42)
        mem = _make_memory(Dict(0x8000 => 0xC9, 0x8001 => 0x42))
        step(s, mem)
        @test (s.P & FLAG_Z) != 0
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_N) == 0
        @test s.A == 0x42
        @test s.cycles == 2
    end

    @testset "CMP A greater sets C clears Z" begin
        s = _state(PC=0x8000, A=0x80)
        mem = _make_memory(Dict(0x8000 => 0xC9, 0x8001 => 0x40))
        step(s, mem)
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_Z) == 0
        @test (s.P & FLAG_N) == 0
    end

    @testset "CMP A less clears C sets N" begin
        s = _state(PC=0x8000, A=0x10)
        mem = _make_memory(Dict(0x8000 => 0xC9, 0x8001 => 0x20))
        step(s, mem)
        @test (s.P & FLAG_C) == 0
        @test (s.P & FLAG_Z) == 0
        @test (s.P & FLAG_N) != 0
    end

    @testset "CMP abs,X page cross adds cycle" begin
        s = _state(PC=0x8000, A=0x10, X=0x10)
        mem = _make_memory(Dict(0x8000 => 0xDD, 0x8001 => 0xF5, 0x8002 => 0x12, 0x1305 => 0x10))
        step(s, mem)
        @test (s.P & FLAG_Z) != 0
        @test s.cycles == 5
    end

    @testset "CPX zero page equal" begin
        s = _state(PC=0x8000, X=0xFF)
        mem = _make_memory(Dict(0x8000 => 0xE4, 0x8001 => 0x10, 0x0010 => 0xFF))
        step(s, mem)
        @test (s.P & FLAG_Z) != 0
        @test (s.P & FLAG_C) != 0
    end

    @testset "CPY immediate Y less" begin
        s = _state(PC=0x8000, Y=0x05)
        mem = _make_memory(Dict(0x8000 => 0xC0, 0x8001 => 0x10))
        step(s, mem)
        @test (s.P & FLAG_C) == 0
        @test (s.P & FLAG_N) != 0
    end

    @testset "BIT zp Z from A AND operand" begin
        s = _state(PC=0x8000, A=0x0F)
        mem = _make_memory(Dict(0x8000 => 0x24, 0x8001 => 0x10, 0x0010 => 0xF0))
        step(s, mem)
        @test (s.P & FLAG_Z) != 0
        @test (s.P & FLAG_N) != 0
        @test (s.P & FLAG_V) != 0
        @test s.A == 0x0F
        @test s.cycles == 3
    end

    @testset "BIT abs clears V when operand bit 6 clear" begin
        s = _state(PC=0x8000, A=0xFF, P=FLAG_U | FLAG_V)
        mem = _make_memory(Dict(0x8000 => 0x2C, 0x8001 => 0x00, 0x8002 => 0x12, 0x1200 => 0x80))
        step(s, mem)
        @test (s.P & FLAG_N) != 0
        @test (s.P & FLAG_V) == 0
        @test (s.P & FLAG_Z) == 0
        @test s.cycles == 4
    end

end

@testset "JuTari P1b2 ADC / SBC binary and BCD" begin

    # ADC binary
    @testset "ADC binary simple" begin
        s = _state(PC=0x8000, A=0x10, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x05))
        step(s, mem)
        @test s.A == 0x15
        @test (s.P & FLAG_C) == 0
        @test (s.P & FLAG_V) == 0
        @test (s.P & FLAG_Z) == 0
        @test (s.P & FLAG_N) == 0
        @test s.cycles == 2
    end

    @testset "ADC binary uses carry in" begin
        s = _state(PC=0x8000, A=0x10, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x05))
        step(s, mem)
        @test s.A == 0x16
    end

    @testset "ADC binary carry out on overflow" begin
        s = _state(PC=0x8000, A=0xFE, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x01))
        step(s, mem)
        @test s.A == 0x00
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_Z) != 0
    end

    @testset "ADC binary positive→negative signed overflow" begin
        s = _state(PC=0x8000, A=0x7F, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x01))
        step(s, mem)
        @test s.A == 0x80
        @test (s.P & FLAG_V) != 0
        @test (s.P & FLAG_N) != 0
        @test (s.P & FLAG_C) == 0
    end

    @testset "ADC binary negative→positive signed overflow" begin
        s = _state(PC=0x8000, A=0x80, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x80))
        step(s, mem)
        @test s.A == 0x00
        @test (s.P & FLAG_V) != 0
        @test (s.P & FLAG_C) != 0
    end

    @testset "ADC binary different signs → no overflow" begin
        s = _state(PC=0x8000, A=0x7F, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x80))
        step(s, mem)
        @test s.A == 0xFF
        @test (s.P & FLAG_V) == 0
        @test (s.P & FLAG_N) != 0
    end

    # ADC decimal
    @testset "ADC decimal simple" begin
        s = _state(PC=0x8000, A=0x12, P=FLAG_U | FLAG_D)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x34))
        step(s, mem)
        @test s.A == 0x46
        @test (s.P & FLAG_C) == 0
    end

    @testset "ADC decimal carry in" begin
        s = _state(PC=0x8000, A=0x12, P=FLAG_U | FLAG_D | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x34))
        step(s, mem)
        @test s.A == 0x47
    end

    @testset "ADC decimal overflow sets carry" begin
        s = _state(PC=0x8000, A=0x55, P=FLAG_U | FLAG_D)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x55))
        step(s, mem)
        @test s.A == 0x10
        @test (s.P & FLAG_C) != 0
    end

    @testset "ADC decimal 99 + 0 + 1 = 100 → 00 with carry" begin
        s = _state(PC=0x8000, A=0x99, P=FLAG_U | FLAG_D | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x69, 0x8001 => 0x00))
        step(s, mem)
        @test s.A == 0x00
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_Z) != 0
    end

    # SBC binary
    @testset "SBC binary simple with carry set" begin
        s = _state(PC=0x8000, A=0x10, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0xE9, 0x8001 => 0x05))
        step(s, mem)
        @test s.A == 0x0B
        @test (s.P & FLAG_C) != 0
    end

    @testset "SBC binary borrow when carry clear" begin
        s = _state(PC=0x8000, A=0x10, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0xE9, 0x8001 => 0x05))
        step(s, mem)
        @test s.A == 0x0A
        @test (s.P & FLAG_C) != 0
    end

    @testset "SBC binary borrow clears carry on underflow" begin
        s = _state(PC=0x8000, A=0x05, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0xE9, 0x8001 => 0x10))
        step(s, mem)
        @test s.A == 0xF5
        @test (s.P & FLAG_C) == 0
    end

    @testset "SBC binary signed overflow" begin
        s = _state(PC=0x8000, A=0x50, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0xE9, 0x8001 => 0xB0))
        step(s, mem)
        @test s.A == 0xA0
        @test (s.P & FLAG_V) != 0
    end

    @testset "Undocumented 0xEB immediate aliases SBC" begin
        s = _state(PC=0x8000, A=0x42, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0xEB, 0x8001 => 0x02))
        step(s, mem)
        @test s.A == 0x40
    end

    # SBC decimal
    @testset "SBC decimal simple with carry set" begin
        s = _state(PC=0x8000, A=0x46, P=FLAG_U | FLAG_D | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0xE9, 0x8001 => 0x12))
        step(s, mem)
        @test s.A == 0x34
        @test (s.P & FLAG_C) != 0
    end

    @testset "SBC decimal borrow into next digit" begin
        s = _state(PC=0x8000, A=0x40, P=FLAG_U | FLAG_D | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0xE9, 0x8001 => 0x12))
        step(s, mem)
        @test s.A == 0x28
    end

    @testset "SBC decimal underflow wraps and clears carry" begin
        s = _state(PC=0x8000, A=0x10, P=FLAG_U | FLAG_D | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0xE9, 0x8001 => 0x20))
        step(s, mem)
        @test s.A == 0x90
        @test (s.P & FLAG_C) == 0
    end

    # Addressing-mode smoke
    @testset "ADC abs,X page cross adds cycle" begin
        s = _state(PC=0x8000, A=0x01, X=0x10, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x7D, 0x8001 => 0xF5, 0x8002 => 0x12, 0x1305 => 0x02))
        step(s, mem)
        @test s.A == 0x03
        @test s.cycles == 5
    end

    @testset "SBC (zp),Y no page cross" begin
        s = _state(PC=0x8000, A=0x10, Y=0x01, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(
            0x8000 => 0xF1, 0x8001 => 0x10,
            0x0010 => 0x00, 0x0011 => 0x12,
            0x1201 => 0x05,
        ))
        step(s, mem)
        @test s.A == 0x0B
        @test s.cycles == 5
    end

end

@testset "JuTari P1c shifts and rotates" begin

    @testset "ASL A shifts bit 7 into carry" begin
        s = _state(PC=0x8000, A=0x80, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x0A))
        step(s, mem)
        @test s.A == 0x00
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_Z) != 0
        @test (s.P & FLAG_N) == 0
        @test s.cycles == 2
    end

    @testset "ASL A normal shift" begin
        s = _state(PC=0x8000, A=0x01, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x0A))
        step(s, mem)
        @test s.A == 0x02
        @test (s.P & FLAG_C) == 0
    end

    @testset "ASL zp writes back and sets N" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0x06, 0x8001 => 0x10, 0x0010 => 0x41))
        step(s, mem)
        @test mem[0x0010 + 1] == 0x82
        @test (s.P & FLAG_N) != 0
        @test (s.P & FLAG_C) == 0
        @test s.cycles == 5
    end

    @testset "ASL abs,X has no page cross penalty" begin
        s = _state(PC=0x8000, X=0x10)
        mem = _make_memory(Dict(0x8000 => 0x1E, 0x8001 => 0xF5, 0x8002 => 0x12, 0x1305 => 0x40))
        step(s, mem)
        @test mem[0x1305 + 1] == 0x80
        @test s.cycles == 7
    end

    @testset "LSR A clears N even when input has bit 7" begin
        s = _state(PC=0x8000, A=0xFF, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x4A))
        step(s, mem)
        @test s.A == 0x7F
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_N) == 0
    end

    @testset "LSR A bit 0 → C, result 0 → Z" begin
        s = _state(PC=0x8000, A=0x01, P=FLAG_U | FLAG_N)
        mem = _make_memory(Dict(0x8000 => 0x4A))
        step(s, mem)
        @test s.A == 0x00
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_Z) != 0
        @test (s.P & FLAG_N) == 0
    end

    @testset "LSR zp,X writes back" begin
        s = _state(PC=0x8000, X=0x05)
        mem = _make_memory(Dict(0x8000 => 0x56, 0x8001 => 0x10, 0x0015 => 0x08))
        step(s, mem)
        @test mem[0x0015 + 1] == 0x04
        @test s.cycles == 6
    end

    @testset "ROL A brings in carry to bit 0" begin
        s = _state(PC=0x8000, A=0x40, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x2A))
        step(s, mem)
        @test s.A == 0x81
        @test (s.P & FLAG_C) == 0
        @test (s.P & FLAG_N) != 0
    end

    @testset "ROL A bit 7 → C" begin
        s = _state(PC=0x8000, A=0x80, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x2A))
        step(s, mem)
        @test s.A == 0x00
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_Z) != 0
    end

    @testset "ROL abs writes back" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x2E, 0x8001 => 0x00, 0x8002 => 0x20, 0x2000 => 0x55))
        step(s, mem)
        @test mem[0x2000 + 1] == 0xAB
        @test (s.P & FLAG_C) == 0
        @test (s.P & FLAG_N) != 0
        @test s.cycles == 6
    end

    @testset "ROR A brings carry to bit 7" begin
        s = _state(PC=0x8000, A=0x02, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x6A))
        step(s, mem)
        @test s.A == 0x81
        @test (s.P & FLAG_C) == 0
        @test (s.P & FLAG_N) != 0
    end

    @testset "ROR A bit 0 → C, result 0 → Z" begin
        s = _state(PC=0x8000, A=0x01, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x6A))
        step(s, mem)
        @test s.A == 0x00
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_Z) != 0
        @test (s.P & FLAG_N) == 0
    end

    @testset "ROR zp writes back" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x66, 0x8001 => 0x10, 0x0010 => 0x02))
        step(s, mem)
        @test mem[0x0010 + 1] == 0x81
        @test (s.P & FLAG_C) == 0
        @test s.cycles == 5
    end

end
