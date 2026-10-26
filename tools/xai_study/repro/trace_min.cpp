// trace_min.cpp — minimal xitari per-frame RAM+screen dumper for the Paper-2
// stream validation. Same boot and output convention as tools/trace_dump.cpp
// (ALEInterface(rom) + resetGame(): 60 NOOP + 4 RESET [+ starting actions unless
// XI_NO_STARTING_ACTIONS is set], then one JSON line per act()), without the
// CpuDebug tap and the bus-trace hook, so it links against stock xitari.
//
//   trace_min <rom.bin> <actions.txt>   > out.jsonl
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "ale_interface.hpp"
using namespace ale;

static void hex(const unsigned char *p, size_t n, std::string &out) {
    static const char *d = "0123456789abcdef";
    out.resize(2 * n);
    for (size_t i = 0; i < n; i++) { out[2 * i] = d[p[i] >> 4]; out[2 * i + 1] = d[p[i] & 15]; }
}

int main(int argc, char **argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s rom actions\n", argv[0]); return 2; }
    std::vector<int> actions;
    std::ifstream f(argv[2]);
    for (std::string line; std::getline(f, line);)
        if (!line.empty() && line[0] != '#') actions.push_back(std::stoi(line));
    ALEInterface ale(argv[1]);
    ale.resetGame();
    std::string ram, scr;
    const ALERAM &r0 = ale.getRAM();
    hex(r0.array(), r0.size(), ram);
    std::printf("{\"frame\":0,\"action\":-1,\"ram\":\"%s\"}\n", ram.c_str());
    int k = 0;
    for (int a : actions) {
        ale.act(static_cast<Action>(a));
        const ALERAM &r = ale.getRAM();
        hex(r.array(), r.size(), ram);
        const ALEScreen &s = ale.getScreen();
        const std::vector<pixel_t> &px = s.getArray();
        hex(reinterpret_cast<const unsigned char *>(px.data()), px.size() * sizeof(pixel_t), scr);
        std::printf("{\"frame\":%d,\"action\":%d,\"game_over\":%s,\"h\":%d,\"w\":%d,\"ram\":\"%s\",\"screen\":\"%s\"}\n",
                    ++k, a, ale.gameOver() ? "true" : "false", s.height(), s.width(),
                    ram.c_str(), scr.c_str());
    }
    return 0;
}
