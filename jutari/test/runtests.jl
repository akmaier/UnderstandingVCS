using Test
using JuTari

@testset "JuTari P0 scaffolding" begin

    @testset "package version" begin
        @test JuTari.VERSION_STRING == "0.0.1"
    end

    @testset "initial CPU state matches RESET convention" begin
        s = initial_cpu_state()
        @test s.SP == 0xFD
        @test (s.P & 0x04) != 0   # I flag set
        @test (s.P & 0x10) != 0   # B flag set
        @test s.cycles == 0
    end

    @testset "step stub advances PC and cycles" begin
        memory = zeros(UInt8, 1 << 16)
        memory[0x0001] = 0xEA   # NOP at $0000 (Julia 1-based index)
        s = initial_cpu_state()
        step(s, memory)
        @test s.PC == 0x0001
        @test s.cycles == 2     # base cycles for NOP
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
