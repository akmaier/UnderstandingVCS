using Test
using JuTari

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
