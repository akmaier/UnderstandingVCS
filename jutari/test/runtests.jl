using Test
using JuTari
using JuTari.CPU: step          # qualified — avoids Base.step collision
using JuTari.Bus: peek, poke!   # qualified — avoids Base.peek collision
using JuTari.TIA: tia_peek, tia_poke!, tia_advance!, tia_apply_wsync!,
                  NTSC_CPU_CYCLES_PER_SCANLINE, NTSC_SCANLINES_PER_FRAME,
                  NUM_REGISTERS, SCREEN_WIDTH, SCREEN_HEIGHT,
                  W_COLUBK, W_WSYNC

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

@testset "JuTari P1d branches, JMP, JSR/RTS" begin

    @testset "BEQ not taken when Z clear" begin
        s = _state(PC=0x8000, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0xF0, 0x8001 => 0x10))
        step(s, mem)
        @test s.PC == 0x8002
        @test s.cycles == 2
    end

    @testset "BEQ taken forward, no page cross" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_Z)
        mem = _make_memory(Dict(0x8000 => 0xF0, 0x8001 => 0x10))
        step(s, mem)
        @test s.PC == 0x8012
        @test s.cycles == 3
    end

    @testset "BEQ taken forward with page cross" begin
        # base = 0x80F2, target = 0x8102 → pages 0x80 vs 0x81
        s = _state(PC=0x80F0, P=FLAG_U | FLAG_Z)
        mem = _make_memory(Dict(0x80F0 => 0xF0, 0x80F1 => 0x10))
        step(s, mem)
        @test s.PC == 0x8102
        @test s.cycles == 4
    end

    @testset "BEQ taken backward with page cross" begin
        # base = 0x8004, offset = -16, target = 0x7FF4
        s = _state(PC=0x8002, P=FLAG_U | FLAG_Z)
        mem = _make_memory(Dict(0x8002 => 0xF0, 0x8003 => 0xF0))
        step(s, mem)
        @test s.PC == 0x7FF4
        @test s.cycles == 4
    end

    @testset "BEQ taken backward within page" begin
        s = _state(PC=0x8100, P=FLAG_U | FLAG_Z)
        mem = _make_memory(Dict(0x8100 => 0xF0, 0x8101 => 0xFE))
        step(s, mem)
        @test s.PC == 0x8100
        @test s.cycles == 3
    end

    @testset "BNE taken when Z clear" begin
        s = _state(PC=0x8000, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0xD0, 0x8001 => 0x05))
        step(s, mem)
        @test s.PC == 0x8007
        @test s.cycles == 3
    end

    @testset "BMI taken when N set" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_N)
        mem = _make_memory(Dict(0x8000 => 0x30, 0x8001 => 0x04))
        step(s, mem)
        @test s.PC == 0x8006
    end

    @testset "BPL not taken when N set" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_N)
        mem = _make_memory(Dict(0x8000 => 0x10, 0x8001 => 0x04))
        step(s, mem)
        @test s.PC == 0x8002
        @test s.cycles == 2
    end

    @testset "BCS taken when C set" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0xB0, 0x8001 => 0x02))
        step(s, mem)
        @test s.PC == 0x8004
    end

    @testset "BVC taken when V clear" begin
        s = _state(PC=0x8000, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x50, 0x8001 => 0x02))
        step(s, mem)
        @test s.PC == 0x8004
    end

    @testset "JMP absolute" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0x4C, 0x8001 => 0x34, 0x8002 => 0x12))
        step(s, mem)
        @test s.PC == 0x1234
        @test s.cycles == 3
    end

    @testset "JMP indirect normal pointer" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(
            0x8000 => 0x6C, 0x8001 => 0x00, 0x8002 => 0x30,
            0x3000 => 0xCD, 0x3001 => 0xAB,
        ))
        step(s, mem)
        @test s.PC == 0xABCD
        @test s.cycles == 5
    end

    @testset "JMP indirect page-wrap bug" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(
            0x8000 => 0x6C, 0x8001 => 0xFF, 0x8002 => 0x30,
            0x30FF => 0xCD,
            0x3000 => 0xAB,
            0x3100 => 0x99,
        ))
        step(s, mem)
        @test s.PC == 0xABCD
    end

    @testset "JSR pushes return addr and jumps" begin
        s = _state(PC=0x8000, SP=0xFD)
        mem = _make_memory(Dict(0x8000 => 0x20, 0x8001 => 0x00, 0x8002 => 0x30))
        step(s, mem)
        @test s.PC == 0x3000
        @test s.SP == 0xFB
        @test mem[0x01FD + 1] == 0x80   # high
        @test mem[0x01FC + 1] == 0x02   # low
        @test s.cycles == 6
    end

    @testset "RTS pops and advances" begin
        s = _state(PC=0x3050, SP=0xFB)
        mem = _make_memory(Dict(
            0x3050 => 0x60,
            0x01FC => 0x02,
            0x01FD => 0x80,
        ))
        step(s, mem)
        @test s.PC == 0x8003
        @test s.SP == 0xFD
        @test s.cycles == 6
    end

    @testset "JSR then RTS round-trip" begin
        s = _state(PC=0x8000, SP=0xFD)
        mem = _make_memory(Dict(
            0x8000 => 0x20, 0x8001 => 0x00, 0x8002 => 0x30,
            0x3000 => 0x60,
        ))
        step(s, mem); @test s.PC == 0x3000
        step(s, mem); @test s.PC == 0x8003
        @test s.SP == 0xFD
    end

end

@testset "JuTari P1e stack push/pull, status flags, NOP" begin

    @testset "PHA pushes A and decrements SP" begin
        s = _state(PC=0x8000, A=0x42, SP=0xFD)
        mem = _make_memory(Dict(0x8000 => 0x48))
        step(s, mem)
        @test s.SP == 0xFC
        @test mem[0x01FD + 1] == 0x42
        @test s.PC == 0x8001
        @test s.cycles == 3
    end

    @testset "PLA sets ZN and increments SP" begin
        s = _state(PC=0x8000, A=0x00, SP=0xFC)
        mem = _make_memory(Dict(0x8000 => 0x68, 0x01FD => 0x80))
        step(s, mem)
        @test s.A == 0x80
        @test s.SP == 0xFD
        @test (s.P & FLAG_N) != 0
        @test (s.P & FLAG_Z) == 0
        @test s.cycles == 4
    end

    @testset "PLA zero sets Z" begin
        s = _state(PC=0x8000, A=0xFF, SP=0xFC)
        mem = _make_memory(Dict(0x8000 => 0x68, 0x01FD => 0x00))
        step(s, mem)
        @test s.A == 0x00
        @test (s.P & FLAG_Z) != 0
    end

    @testset "PHA / PLA round-trip preserves A" begin
        s = _state(PC=0x8000, A=0x77, SP=0xFD)
        mem = _make_memory(Dict(0x8000 => 0x48, 0x8001 => 0xA9, 0x8002 => 0x00, 0x8003 => 0x68))
        step(s, mem); step(s, mem); step(s, mem)
        @test s.A == 0x77
        @test s.SP == 0xFD
    end

    @testset "PHP pushes P with B and U set" begin
        s = _state(PC=0x8000, SP=0xFD, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x08))
        step(s, mem)
        pushed = mem[0x01FD + 1]
        @test (pushed & FLAG_B) != 0
        @test (pushed & FLAG_U) != 0
        @test (pushed & FLAG_C) != 0
        @test s.SP == 0xFC
        @test s.cycles == 3
    end

    @testset "PLP forces B and U on pull" begin
        s = _state(PC=0x8000, SP=0xFC, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x28, 0x01FD => FLAG_N | FLAG_Z))
        step(s, mem)
        @test (s.P & FLAG_N) != 0
        @test (s.P & FLAG_Z) != 0
        @test (s.P & FLAG_B) != 0
        @test (s.P & FLAG_U) != 0
        @test s.SP == 0xFD
        @test s.cycles == 4
    end

    @testset "PHP / PLP round-trip with intervening clobber" begin
        original_p = FLAG_U | FLAG_C | FLAG_V | FLAG_N
        s = _state(PC=0x8000, SP=0xFD, P=original_p)
        mem = _make_memory(Dict(0x8000 => 0x08, 0x8001 => 0xA9, 0x8002 => 0x00, 0x8003 => 0x28))
        step(s, mem); step(s, mem); step(s, mem)
        @test s.P == (original_p | FLAG_B)
        @test s.SP == 0xFD
    end

    @testset "SEC sets carry" begin
        s = _state(PC=0x8000, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x38))
        step(s, mem)
        @test (s.P & FLAG_C) != 0
        @test s.cycles == 2
    end

    @testset "CLC clears carry" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(0x8000 => 0x18))
        step(s, mem)
        @test (s.P & FLAG_C) == 0
    end

    @testset "SEI sets I" begin
        s = _state(PC=0x8000, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0x78))
        step(s, mem)
        @test (s.P & FLAG_I) != 0
    end

    @testset "CLI clears I" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_I)
        mem = _make_memory(Dict(0x8000 => 0x58))
        step(s, mem)
        @test (s.P & FLAG_I) == 0
    end

    @testset "SED sets D" begin
        s = _state(PC=0x8000, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0xF8))
        step(s, mem)
        @test (s.P & FLAG_D) != 0
    end

    @testset "CLD clears D" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_D)
        mem = _make_memory(Dict(0x8000 => 0xD8))
        step(s, mem)
        @test (s.P & FLAG_D) == 0
    end

    @testset "CLV clears V" begin
        s = _state(PC=0x8000, P=FLAG_U | FLAG_V)
        mem = _make_memory(Dict(0x8000 => 0xB8))
        step(s, mem)
        @test (s.P & FLAG_V) == 0
    end

    @testset "NOP advances PC and cycles" begin
        s = _state(PC=0x8000, P=FLAG_U)
        mem = _make_memory(Dict(0x8000 => 0xEA))
        step(s, mem)
        @test s.PC == 0x8001
        @test s.cycles == 2
        @test s.P == FLAG_U
    end

end

@testset "JuTari P1f INC/DEC, INX/INY/DEX/DEY, BRK/RTI" begin

    @testset "INC zp increments and writes back" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xE6, 0x8001 => 0x10, 0x0010 => 0x41))
        step(s, mem)
        @test mem[0x0010 + 1] == 0x42
        @test (s.P & FLAG_Z) == 0
        @test (s.P & FLAG_N) == 0
        @test s.cycles == 5
    end

    @testset "INC wraps FF→00 and sets Z" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xE6, 0x8001 => 0x10, 0x0010 => 0xFF))
        step(s, mem)
        @test mem[0x0010 + 1] == 0x00
        @test (s.P & FLAG_Z) != 0
    end

    @testset "INC abs,X sets N" begin
        s = _state(PC=0x8000, X=0x01)
        mem = _make_memory(Dict(0x8000 => 0xFE, 0x8001 => 0x00, 0x8002 => 0x12, 0x1201 => 0x7F))
        step(s, mem)
        @test mem[0x1201 + 1] == 0x80
        @test (s.P & FLAG_N) != 0
        @test s.cycles == 7
    end

    @testset "DEC zp" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xC6, 0x8001 => 0x10, 0x0010 => 0x01))
        step(s, mem)
        @test mem[0x0010 + 1] == 0x00
        @test (s.P & FLAG_Z) != 0
    end

    @testset "DEC wraps 00→FF sets N" begin
        s = _state(PC=0x8000)
        mem = _make_memory(Dict(0x8000 => 0xC6, 0x8001 => 0x10, 0x0010 => 0x00))
        step(s, mem)
        @test mem[0x0010 + 1] == 0xFF
        @test (s.P & FLAG_N) != 0
    end

    @testset "INX normal" begin
        s = _state(PC=0x8000, X=0x05)
        mem = _make_memory(Dict(0x8000 => 0xE8))
        step(s, mem)
        @test s.X == 0x06
        @test s.cycles == 2
    end

    @testset "INX wraps and sets Z" begin
        s = _state(PC=0x8000, X=0xFF)
        mem = _make_memory(Dict(0x8000 => 0xE8))
        step(s, mem)
        @test s.X == 0x00
        @test (s.P & FLAG_Z) != 0
    end

    @testset "INY sets N" begin
        s = _state(PC=0x8000, Y=0x7F)
        mem = _make_memory(Dict(0x8000 => 0xC8))
        step(s, mem)
        @test s.Y == 0x80
        @test (s.P & FLAG_N) != 0
    end

    @testset "DEX wraps and sets N" begin
        s = _state(PC=0x8000, X=0x00)
        mem = _make_memory(Dict(0x8000 => 0xCA))
        step(s, mem)
        @test s.X == 0xFF
        @test (s.P & FLAG_N) != 0
    end

    @testset "DEY to zero sets Z" begin
        s = _state(PC=0x8000, Y=0x01)
        mem = _make_memory(Dict(0x8000 => 0x88))
        step(s, mem)
        @test s.Y == 0x00
        @test (s.P & FLAG_Z) != 0
    end

    @testset "BRK pushes PC+2 and jumps via vector" begin
        s = _state(PC=0x8000, SP=0xFD, P=FLAG_U)
        mem = _make_memory(Dict(
            0x8000 => 0x00,
            0xFFFE => 0x34, 0xFFFF => 0x12,
        ))
        step(s, mem)
        @test mem[0x01FD + 1] == 0x80
        @test mem[0x01FC + 1] == 0x02
        pushed_p = mem[0x01FB + 1]
        @test (pushed_p & FLAG_B) != 0
        @test (pushed_p & FLAG_U) != 0
        @test (s.P & FLAG_I) != 0
        @test s.PC == 0x1234
        @test s.SP == 0xFA
        @test s.cycles == 7
    end

    @testset "RTI pops P then PC (no +1)" begin
        s = _state(PC=0x1234, SP=0xFA)
        mem = _make_memory(Dict(
            0x1234 => 0x40,
            0x01FB => FLAG_B | FLAG_U | FLAG_C,
            0x01FC => 0x02,
            0x01FD => 0x80,
        ))
        step(s, mem)
        @test s.PC == 0x8002
        @test (s.P & FLAG_C) != 0
        @test (s.P & FLAG_B) != 0
        @test (s.P & FLAG_U) != 0
        @test s.SP == 0xFD
        @test s.cycles == 6
    end

    @testset "BRK then RTI round-trip" begin
        s = _state(PC=0x8000, SP=0xFD, P=FLAG_U | FLAG_C)
        mem = _make_memory(Dict(
            0x8000 => 0x00,
            0x1234 => 0x40,
            0xFFFE => 0x34, 0xFFFF => 0x12,
        ))
        step(s, mem); @test s.PC == 0x1234
        step(s, mem); @test s.PC == 0x8002
        @test s.SP == 0xFD
        @test (s.P & FLAG_C) != 0
    end

end

@testset "JuTari P2 6507 bus + address decode" begin

    @testset "RAM read/write at canonical address" begin
        bus = initial_bus()
        poke!(bus, 0x0080, 0x42)
        @test peek(bus, 0x0080) == 0x42
        @test bus.ram[1] == 0x42
    end

    @testset "RAM mirror at stack page (\$0180 aliases \$0080)" begin
        bus = initial_bus()
        poke!(bus, 0x0180, 0x55)
        @test peek(bus, 0x0080) == 0x55
        @test peek(bus, 0x0180) == 0x55
    end

    @testset "RAM uses low 7 bits of address" begin
        bus = initial_bus()
        poke!(bus, 0x0098, 0xAB)
        @test peek(bus, 0x0098) == 0xAB
        @test peek(bus, 0x0198) == 0xAB
        @test peek(bus, 0x0498) == 0xAB
    end

    @testset "13-bit mirror — high addresses wrap" begin
        rom = zeros(UInt8, 4096); rom[0x100 + 1] = 0xEE
        bus = initial_bus(rom)
        @test peek(bus, 0x1100) == 0xEE
        @test peek(bus, 0xF100) == 0xEE
        @test peek(bus, 0x9100) == 0xEE
    end

    @testset "TIA region reads zero and writes ignored" begin
        bus = initial_bus()
        @test peek(bus, 0x0000) == 0
        @test peek(bus, 0x007F) == 0
        ram_before = copy(bus.ram)
        poke!(bus, 0x0001, 0xFF)
        @test bus.ram == ram_before
    end

    @testset "RIOT I/O region reads zero and writes ignored" begin
        bus = initial_bus()
        @test peek(bus, 0x0280) == 0
        @test peek(bus, 0x029F) == 0
        ram_before = copy(bus.ram)
        poke!(bus, 0x0284, 0xFF)
        @test bus.ram == ram_before
    end

    @testset "ROM is read-only" begin
        rom = fill(UInt8(0xAA), 4096)
        bus = initial_bus(rom)
        @test peek(bus, 0x1000) == 0xAA
        poke!(bus, 0x1000, 0x55)
        @test peek(bus, 0x1000) == 0xAA
    end

    @testset "Rejects non-4K ROM" begin
        @test_throws ArgumentError initial_bus(zeros(UInt8, 8192))
    end

    @testset "LDA #imm via bus from \$F000" begin
        rom = zeros(UInt8, 4096); rom[1] = 0xA9; rom[2] = 0x42
        bus = initial_bus(rom)
        s = _state(PC=0xF000)
        step(s, bus)
        @test s.A == 0x42
        @test s.PC == 0xF002
        @test s.cycles == 2
    end

    @testset "STA then LDA round-trips through RAM" begin
        rom = zeros(UInt8, 4096)
        program = [0xA9, 0x77, 0x85, 0x80, 0xA9, 0x00, 0xA5, 0x80]
        for (i, b) in enumerate(program)
            rom[i] = UInt8(b)
        end
        bus = initial_bus(rom)
        s = _state(PC=0xF000)
        for _ in 1:4
            step(s, bus)
        end
        @test s.A == 0x77
        @test bus.ram[1] == 0x77
    end

    @testset "JSR / RTS stack pushes land in RAM via the \$01xx mirror" begin
        rom = zeros(UInt8, 4096)
        rom[1] = 0x20; rom[2] = 0x05; rom[3] = 0xF0   # JSR $F005 at $F000
        rom[6] = 0x60                                  # RTS at $F005
        bus = initial_bus(rom)
        s = _state(PC=0xF000, SP=0xFD)
        step(s, bus)
        @test s.PC == 0xF005
        @test s.SP == 0xFB
        # Pushed return address at $01FD/$01FC = RAM offsets 0x7D/0x7C.
        @test bus.ram[0x7D + 1] == 0xF0   # high
        @test bus.ram[0x7C + 1] == 0x02   # low
        step(s, bus)
        @test s.PC == 0xF003
        @test s.SP == 0xFD
    end

end

@testset "JuTari P3a TIA register file + scanline timing + WSYNC" begin

    @testset "initial TIA state is zeroed" begin
        tia = initial_tia_state()
        @test length(tia.registers) == NUM_REGISTERS
        @test sum(tia.registers) == 0
        @test tia.scanline_cycle == 0
        @test tia.scanline == 0
        @test tia.frame == 0
        @test tia.wsync_pending == false
        @test size(tia.framebuffer) == (SCREEN_HEIGHT, SCREEN_WIDTH)
        @test sum(tia.framebuffer) == 0
    end

    @testset "tia_poke! stores byte in register file" begin
        tia = initial_tia_state()
        tia_poke!(tia, W_COLUBK, 0x1C)
        @test tia.registers[W_COLUBK + 1] == 0x1C
    end

    @testset "tia_poke! WSYNC sets pending flag" begin
        tia = initial_tia_state()
        tia_poke!(tia, W_WSYNC, 0x00)
        @test tia.wsync_pending == true
    end

    @testset "tia_poke! decodes only low 6 bits of address" begin
        tia = initial_tia_state()
        # Address $42 = $40 | $02 → register $02 (WSYNC)
        tia_poke!(tia, 0x42, 0xFF)
        @test tia.wsync_pending == true
    end

    @testset "tia_peek is zero stub in P3a" begin
        tia = initial_tia_state()
        for addr in (0x00, 0x07, 0x08, 0x0D, 0x30, 0x3F)
            @test tia_peek(tia, addr) == 0
        end
    end

    @testset "tia_advance! within scanline" begin
        tia = initial_tia_state()
        tia_advance!(tia, 30)
        @test tia.scanline_cycle == 30
        @test tia.scanline == 0
        @test tia.frame == 0
    end

    @testset "tia_advance! crosses scanline boundary" begin
        tia = initial_tia_state()
        tia_advance!(tia, NTSC_CPU_CYCLES_PER_SCANLINE)
        @test tia.scanline_cycle == 0
        @test tia.scanline == 1
    end

    @testset "tia_advance! crosses multiple scanlines" begin
        tia = initial_tia_state()
        tia_advance!(tia, NTSC_CPU_CYCLES_PER_SCANLINE * 5 + 10)
        @test tia.scanline_cycle == 10
        @test tia.scanline == 5
    end

    @testset "tia_advance! crosses frame boundary" begin
        tia = initial_tia_state()
        tia_advance!(tia, NTSC_CPU_CYCLES_PER_SCANLINE * NTSC_SCANLINES_PER_FRAME)
        @test tia.scanline_cycle == 0
        @test tia.scanline == 0
        @test tia.frame == 1
    end

    @testset "tia_apply_wsync! noop when no pending" begin
        tia = initial_tia_state()
        stall = tia_apply_wsync!(tia)
        @test stall == 0
    end

    @testset "tia_apply_wsync! stalls to next scanline boundary" begin
        tia = initial_tia_state()
        tia.scanline_cycle = 20
        tia.wsync_pending = true
        stall = tia_apply_wsync!(tia)
        @test stall == 56                       # 76 - 20
        @test tia.scanline_cycle == 0
        @test tia.scanline == 1
        @test tia.wsync_pending == false
    end

    @testset "tia_apply_wsync! at scanline boundary is zero-cycle" begin
        tia = initial_tia_state()
        tia.wsync_pending = true                # scanline_cycle already 0
        stall = tia_apply_wsync!(tia)
        @test stall == 0
        @test tia.scanline == 0
        @test tia.wsync_pending == false
    end

    @testset "bus TIA write records in register file" begin
        bus = initial_bus()
        poke!(bus, 0x0009, 0x42)                # COLUBK = 0x42
        @test bus.tia.registers[W_COLUBK + 1] == 0x42
    end

    @testset "bus TIA write to WSYNC sets pending flag" begin
        bus = initial_bus()
        poke!(bus, 0x0002, 0x00)
        @test bus.tia.wsync_pending == true
    end

    @testset "step advances tia.scanline_cycle by instruction cycles" begin
        rom = zeros(UInt8, 4096); rom[1] = 0xEA   # NOP at $F000
        bus = initial_bus(rom)
        s = _state(PC=0xF000)
        step(s, bus)
        @test bus.tia.scanline_cycle == 2
        @test bus.tia.scanline == 0
    end

    @testset "step crosses scanline when enough cycles accumulate" begin
        rom = fill(UInt8(0xEA), 4096)             # all NOPs
        bus = initial_bus(rom)
        s = _state(PC=0xF000)
        for _ in 1:38                             # 76 / 2 = 38 NOPs to fill a scanline
            step(s, bus)
        end
        @test bus.tia.scanline_cycle == 0
        @test bus.tia.scanline == 1
    end

    @testset "STA WSYNC stalls CPU to next scanline" begin
        rom = zeros(UInt8, 4096)
        rom[1] = 0xA9; rom[2] = 0x00              # LDA #$00 (2 cyc)
        rom[3] = 0x85; rom[4] = 0x02              # STA $02  (3 cyc) → WSYNC
        bus = initial_bus(rom)
        s = _state(PC=0xF000)
        step(s, bus); @test s.cycles == 2
        step(s, bus)
        @test s.cycles == NTSC_CPU_CYCLES_PER_SCANLINE
        @test bus.tia.scanline == 1
        @test bus.tia.scanline_cycle == 0
        @test bus.tia.wsync_pending == false
    end

    @testset "flat memory does not trigger TIA post-step" begin
        mem = zeros(UInt8, 1 << 16); mem[1] = 0xEA    # NOP at $0000
        s = _state(PC=0x0000)
        step(s, mem)
        @test s.cycles == 2
        @test isa(mem, Vector{UInt8})                  # still flat, not a Bus
    end

end
