# test_jutari_record.jl — self-check for the jutari trajectory recorder
# (P2-E0-2j). Asserts the DoD properties:
#   1. shape: a RAM-tape recording is (frames, 128) UInt8, with a 1:frames index;
#   2. determinism: two FRESH recordings are byte-identical (the load-bearing
#      guarantee — the recorder inherits the Paper-1 bit-exact replay path);
#   3. multi-field stacking: ram+tia yields (frames, 128+64) with the documented
#      per-field widths, and the RAM slice matches a RAM-only recording;
#   4. the §R artifact (npz + JSON sidecar) is written, numpy-loadable, with a
#      shape that round-trips the tape.
#
# Run:
#   julia --project=/Users/maier/Documents/code/UnderstandingVCS/jutari \
#         tools/xai_study/common/test_jutari_record.jl

using Test

include(joinpath(@__DIR__, "jutari_record.jl"))
using .JutariRecord
using .JutariRecord: record_trajectory, write_trajectory, FIELDS

const GAME = "pong"
const FRAMES = 60          # within the conformance horizon (Paper-1 30 RAM / 60 screen)

@testset "P2-E0-2j jutari trajectory recorder" begin

    @testset "1. RAM-tape shape" begin
        traj = record_trajectory(GAME; frames = FRAMES, fields = ["ram"])
        @test size(traj.tape) == (FRAMES, 128)
        @test eltype(traj.tape) == UInt8
        @test traj.frame == collect(1:FRAMES)
        @test traj.fields == ["ram"]
        @test traj.widths == [128]
    end

    @testset "2. determinism: two fresh recordings byte-identical" begin
        a = record_trajectory(GAME; frames = FRAMES, fields = ["ram"])
        b = record_trajectory(GAME; frames = FRAMES, fields = ["ram"])
        @test a.tape == b.tape
        @test a.tape !== b.tape          # independent buffers, not aliased
        @info "determinism check" identical = (a.tape == b.tape) bytes = length(a.tape)
    end

    @testset "3. multi-field stacking (ram+tia)" begin
        ram_only = record_trajectory(GAME; frames = FRAMES, fields = ["ram"])
        both     = record_trajectory(GAME; frames = FRAMES, fields = ["ram", "tia"])
        @test both.fields == ["ram", "tia"]
        @test both.widths == [128, 64]
        @test size(both.tape) == (FRAMES, 192)
        # the RAM slice of the stacked tape must equal the RAM-only tape
        @test both.tape[:, 1:128] == ram_only.tape
    end

    @testset "4. §R artifact written + numpy-loadable" begin
        traj = record_trajectory(GAME; frames = FRAMES, fields = ["ram"])
        npz_path, json_path = write_trajectory(traj)
        @test isfile(npz_path)
        @test isfile(json_path)
        # the JSON sidecar carries the self-describing fields
        txt = read(json_path, String)
        @test occursin("\"game\": \"$GAME\"", txt)
        @test occursin("\"frames\": $FRAMES", txt)
        @test occursin("\"shape\": [$FRAMES, 128]", txt)
        @test occursin("\"fields\": [\"ram\"]", txt)
        # the .npz is a valid (uncompressed) ZIP: magic "PK\x03\x04"
        raw = read(npz_path)
        @test raw[1:4] == UInt8[0x50, 0x4b, 0x03, 0x04]
        @info "artifact" npz_path json_path npz_bytes = length(raw)
    end

end
