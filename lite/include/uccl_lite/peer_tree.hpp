#pragma once
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <vector>

namespace uccl_lite {
struct PeerTreeRoute {
  std::vector<int> parent, depth;
  int height = 0;
};
// Validate a root-zero tree and derive the forwarding schedule for one source.
// Logical ranks remain unchanged; rooting never reorders reduction operands.
inline PeerTreeRoute peerTreeRoute(std::vector<int> const& tree, int source) {
  int n = int(tree.size());
  if (n < 2 || n > 32 || source < 0 || source >= n || tree[0] != -1)
    throw std::invalid_argument("invalid peer tree extent/root");
  std::vector<std::vector<int>> adjacent(n);
  for (int r = 1; r < n; ++r) {
    int p = tree[r];
    if (p < 0 || p >= n || p == r)
      throw std::invalid_argument("invalid peer tree parent");
    adjacent[r].push_back(p);
    adjacent[p].push_back(r);
  }
  PeerTreeRoute result{std::vector<int>(n, -1), std::vector<int>(n, -1), 0};
  std::queue<int> queue;
  result.depth[source] = 0;
  queue.push(source);
  while (!queue.empty()) {
    int r = queue.front();
    queue.pop();
    for (int next : adjacent[r])
      if (result.depth[next] < 0) {
        result.parent[next] = r;
        result.depth[next] = result.depth[r] + 1;
        result.height = std::max(result.height, result.depth[next]);
        queue.push(next);
      }
  }
  if (std::find(result.depth.begin(), result.depth.end(), -1) !=
      result.depth.end())
    throw std::invalid_argument("peer tree disconnected or cyclic");
  return result;
}
}  // namespace uccl_lite
