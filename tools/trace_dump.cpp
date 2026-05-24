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

int main(int argc, char **argv) {
    std::string rom, actions_file;
    int max_frames = 0;
    bool dump_screen = false;
    bool dump_cpu    = false;
    bool repeat_last = false;
    bool auto_reset  = false;

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

    return 0;
}
