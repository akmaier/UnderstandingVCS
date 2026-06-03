// trace_dump.cpp
//
// Frame-level conformance-trace generator for the UnderstandingVCS project.
//
// Drives an ALEInterface (xitari) deterministically with a fixed action stream
// and writes one JSON line per frame to stdout containing:
//
//     {"frame": N, "ep_frame": N, "reward": R, "cum_reward": CR,
//      "lives": L, "done": false, "ram": "<256 hex chars>"}
//
// Optional flags:
//     --screen          adds "screen": "<2*h*w hex chars>"
//     --cpu             adds "cpu": {"A":N,"X":N,"Y":N,"SP":N,"P":N,
//                                    "PC":N,"cycles":N}
//
// The --cpu flag uses the `friend class CpuDebug` declaration that lives in
// xitari/emucore/m6502/src/M6502.hxx (and the matching friend in TIA.hxx /
// future M6532 access) — see the comments next to `_CpuDebug` below for how
// we tap into M6502's protected state without modifying xitari.
//
// Build: see ./Makefile (depends on a built libxitari.a + xitari headers
// living at ../xitari).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "ale_interface.hpp"
#include "OSystem.hxx"
#include "Console.hxx"
#include "System.hxx"
#include "M6502.hxx"

// ----------------------------------------------------------------------- //
// PXC1-x debug taps
// ----------------------------------------------------------------------- //
//
// xitari was structured for a Stella-style debugger: M6502.hxx declares
// `friend class CpuDebug;` and TIA.hxx declares `friend class TIADebug;`.
// xitari itself never defines those classes (Stella does), so we can define
// them here — in the same global namespace the friend declarations name —
// and silently gain access to M6502's protected register file without
// modifying any xitari header.
//
// This is the minimum-disruption path to the PXC1-x round 2+ diagnostics
// described in PORTING_PLAN.md §4.1.

namespace ale {
class CpuDebug {
public:
    static uInt8 a (const M6502& cpu) { return cpu.A;  }
    static uInt8 x (const M6502& cpu) { return cpu.X;  }
    static uInt8 y (const M6502& cpu) { return cpu.Y;  }
    static uInt8 sp(const M6502& cpu) { return cpu.SP; }
    // PS() is a protected method on M6502 returning the packed NV-BDIZC byte.
    static uInt8 p (const M6502& cpu) { return cpu.PS(); }
};
} // namespace ale

using namespace ale;

static void hex_encode(const unsigned char *src, size_t n, std::string &dst) {
    static const char *H = "0123456789abcdef";
    dst.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
        dst[2 * i]     = H[(src[i] >> 4) & 0x0F];
        dst[2 * i + 1] = H[src[i] & 0x0F];
    }
}

static std::vector<int> load_actions(const std::string &path) {
    std::vector<int> out;
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "trace_dump: cannot open actions file %s\n", path.c_str());
        std::exit(2);
    }
    std::string line;
    while (std::getline(f, line)) {
        // strip whitespace and trailing CR; skip blanks and comments
        size_t a = line.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        if (line[a] == '#') continue;
        size_t b = line.find_last_not_of(" \t\r\n");
        out.push_back(std::atoi(line.substr(a, b - a + 1).c_str()));
    }
    return out;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
        "usage: %s --rom <path> --actions <file> [--max-frames N] [--screen]\n"
        "       [--cpu] [--repeat-last-on-exhaust]\n\n"
        "Writes one JSONL line per frame to stdout.\n"
        "Actions file: one integer ALE action per line (see ale_interface.hpp Action enum).\n"
        "--cpu adds a `cpu` object per line with A/X/Y/SP/P/PC + system cycles.\n",
        argv0);
}

// --bus-trace state — written by the System callback (one global).
static FILE *g_bus_trace_fp = nullptr;
static int g_bus_trace_frame = 0;
static long long g_bus_trace_idx = 0;
static int g_bus_trace_min_frame = 0;
static int g_bus_trace_max_frame = INT32_MAX;

static void bus_trace_callback(uInt16 addr, uInt8 value, bool is_write,
                                uInt32 cpu_cycles) {
    if (!g_bus_trace_fp) return;
    if (g_bus_trace_frame < g_bus_trace_min_frame) return;
    if (g_bus_trace_frame > g_bus_trace_max_frame) return;
    // Derive (scanline, scanline_cycle, color_clock) from cpu_cycles assuming
    // xitari resets cycles() per frame. 228 cc/scanline, 76 cy/scanline, 3 cc/cy.
    uInt32 cc_in_frame = cpu_cycles * 3;
    int scanline       = cc_in_frame / 228;
    int sc             = (cc_in_frame % 228) / 3;
    int cc             = cc_in_frame % 228;
    g_bus_trace_idx++;
    std::fprintf(g_bus_trace_fp,
                 "%lld,%d,%s,%d,%d,%d,%x,%u\n",
                 g_bus_trace_idx, g_bus_trace_frame,
                 is_write ? "poke" : "peek",
                 scanline, sc, cc,
                 (unsigned)addr, (unsigned)value);
}

int main(int argc, char **argv) {
    std::string rom, actions_file;
    int max_frames = 0;
    bool dump_screen = false;
    bool dump_cpu    = false;
    bool repeat_last = false;
    bool auto_reset  = false;
    std::string bus_trace_path;
    int bus_trace_min = 0, bus_trace_max = INT32_MAX;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--rom" && i + 1 < argc) { rom = argv[++i]; }
        else if (a == "--actions" && i + 1 < argc) { actions_file = argv[++i]; }
        else if (a == "--max-frames" && i + 1 < argc) { max_frames = std::atoi(argv[++i]); }
        else if (a == "--screen") { dump_screen = true; }
        else if (a == "--cpu") { dump_cpu = true; }
        else if (a == "--repeat-last-on-exhaust") { repeat_last = true; }
        // --auto-reset: when the cart declares gameOver, call
        // resetGame() and keep going. Useful for fixed-length video
        // captures of games that end quickly under random play
        // (e.g. Breakout's 5-life-then-dead default).
        else if (a == "--auto-reset") { auto_reset = true; }
        // --bus-trace PATH: emit per-bus-op CSV alongside the per-frame
        // stdout JSONL. Format matches jutari's tools/cpu_tia_cycle_trace.jl
        // output so the two can be diffed event-by-event.
        // --bus-trace-frames LO,HI: limit which frames get traced (1-based,
        //   inclusive). Useful to avoid 64+ MB traces when only interested
        //   in a specific divergence frame.
        else if (a == "--bus-trace" && i + 1 < argc) { bus_trace_path = argv[++i]; }
        else if (a == "--bus-trace-frames" && i + 1 < argc) {
            std::string arg = argv[++i];
            auto comma = arg.find(',');
            if (comma != std::string::npos) {
                bus_trace_min = std::atoi(arg.substr(0, comma).c_str());
                bus_trace_max = std::atoi(arg.substr(comma + 1).c_str());
            } else {
                bus_trace_min = bus_trace_max = std::atoi(arg.c_str());
            }
        }
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "trace_dump: unknown arg %s\n", a.c_str()); usage(argv[0]); return 2; }
    }

    if (rom.empty() || actions_file.empty()) { usage(argv[0]); return 2; }

    std::vector<int> actions = load_actions(actions_file);
    if (actions.empty()) {
        std::fprintf(stderr, "trace_dump: actions file is empty\n");
        return 2;
    }

    ALEInterface ale(rom);
    ale.resetGame();

    // Set up bus trace BEFORE the action loop (post-reset, so the boot
    // burn isn't traced — matches jutari's `cpu_tia_cycle_trace.jl`
    // which discards trace events before frame 1).
    if (!bus_trace_path.empty()) {
        g_bus_trace_fp = std::fopen(bus_trace_path.c_str(), "w");
        if (!g_bus_trace_fp) {
            std::fprintf(stderr, "trace_dump: cannot open --bus-trace %s\n",
                         bus_trace_path.c_str());
            return 2;
        }
        g_bus_trace_min_frame = bus_trace_min;
        g_bus_trace_max_frame = bus_trace_max;
        std::fprintf(g_bus_trace_fp,
                     "global_idx,frame,kind,scanline,scanline_cycle,color_clock,addr,value\n");
        system_bus_trace_callback = bus_trace_callback;
    }

    long cum_reward = 0;
    size_t a_idx = 0;
    int frame_count = 0;
    std::string ram_hex, screen_hex;

    while (true) {
        if (max_frames > 0 && frame_count >= max_frames) break;
        if (ale.gameOver()) {
            if (auto_reset) {
                ale.resetGame();
            } else {
                break;
            }
        }

        int act_id;
        if (a_idx < actions.size()) {
            act_id = actions[a_idx++];
        } else if (repeat_last) {
            act_id = actions.back();
        } else {
            break;
        }

        g_bus_trace_frame = frame_count + 1;
        reward_t r = ale.act(static_cast<Action>(act_id));
        cum_reward += r;

        const ALERAM &ram = ale.getRAM();
        hex_encode(ram.array(), ram.size(), ram_hex);

        std::fprintf(stdout,
            "{\"frame\":%d,\"ep_frame\":%d,\"action\":%d,\"reward\":%ld,"
            "\"cum_reward\":%ld,\"lives\":%d,\"done\":%s,\"ram\":\"%s\"",
            ale.getFrameNumber(), ale.getEpisodeFrameNumber(), act_id,
            (long)r, cum_reward, ale.lives(),
            ale.gameOver() ? "true" : "false", ram_hex.c_str());

        if (dump_screen) {
            const ALEScreen &scr = ale.getScreen();
            const std::vector<pixel_t> &arr = scr.getArray();
            hex_encode(reinterpret_cast<const unsigned char *>(arr.data()),
                       arr.size() * sizeof(pixel_t), screen_hex);
            std::fprintf(stdout, ",\"h\":%d,\"w\":%d,\"screen\":\"%s\"",
                         scr.height(), scr.width(), screen_hex.c_str());
        }

        if (dump_cpu) {
            // PXC1-x round 2+: dump CPU register state at the end of the
            // frame, via the CpuDebug friend tap defined above.
            const M6502 &cpu = ale.osystem().console().system().m6502();
            std::fprintf(stdout,
                ",\"cpu\":{\"A\":%u,\"X\":%u,\"Y\":%u,\"SP\":%u,"
                "\"P\":%u,\"PC\":%u,\"cycles\":%u}",
                static_cast<unsigned>(CpuDebug::a(cpu)),
                static_cast<unsigned>(CpuDebug::x(cpu)),
                static_cast<unsigned>(CpuDebug::y(cpu)),
                static_cast<unsigned>(CpuDebug::sp(cpu)),
                static_cast<unsigned>(CpuDebug::p(cpu)),
                static_cast<unsigned>(cpu.getPC()),
                static_cast<unsigned>(ale.osystem().console().system().cycles()));
        }

        std::fprintf(stdout, "}\n");
        frame_count++;
    }

    if (g_bus_trace_fp) {
        std::fclose(g_bus_trace_fp);
        system_bus_trace_callback = nullptr;
        std::fprintf(stderr,
                     "trace_dump: wrote %lld bus-op events to %s\n",
                     g_bus_trace_idx, bus_trace_path.c_str());
    }

    return 0;
}
