#!/usr/bin/env python3
"""Adapt NIXL's UCCL plugin to variable-length notify_msg_t.

UCCL #1037 changed notify_msg_t.name / notify_msg_t.msg from char[256] to
std::string. NIXL main still treats those fields as C arrays (strncpy,
memcpy, memset, sizeof(notify_msg_t::msg)), which fails to compile against
current UCCL. Apply this after cloning NIXL until upstream matches.
"""

from __future__ import annotations

import sys
from pathlib import Path

ALREADY_PATCHED = "notify_msg.name = local_agent_name_;"

REPLACEMENTS = (
    (
        """        if (serialized.size() > sizeof(notify_msg_t::msg)) {
            NIXL_ERROR << "Notification message too large: " << serialized.size()
                       << " bytes, max: " << sizeof(notify_msg_t::msg) << " bytes";
        } else {
            notify_msg_t notify_msg = {};
            strncpy(notify_msg.name, local_agent_name_.c_str(), sizeof(notify_msg.name) - 1);
            memcpy(notify_msg.msg, serialized.c_str(), serialized.size());

            int result = uccl_engine_send_notif(conn, &notify_msg);
            if (result < 0) {
                NIXL_ERROR << "Failed to send notify message";
                return NIXL_ERR_BACKEND;
            }
            NIXL_DEBUG << "Transfer complete, sent notification: " << uccl_handle->notif_msg;
        }
""",
        """        notify_msg_t notify_msg;
        notify_msg.name = local_agent_name_;
        notify_msg.msg = serialized;

        int result = uccl_engine_send_notif(conn, &notify_msg);
        if (result < 0) {
            NIXL_ERROR << "Failed to send notify message";
            return NIXL_ERR_BACKEND;
        }
        NIXL_DEBUG << "Transfer complete, sent notification: " << uccl_handle->notif_msg;
""",
    ),
    (
        """        size_t msg_len = sizeof(notify_msgs[i].msg);
        std::string serialized_str(notify_msgs[i].msg, msg_len);
""",
        """        std::string serialized_str = notify_msgs[i].msg;
""",
    ),
    (
        """    if (serialized.size() > sizeof(notify_msg_t::msg)) {
        NIXL_ERROR << "Notification message too large: " << serialized.size()
                   << " bytes, max: " << sizeof(notify_msg_t::msg) << " bytes";
        return NIXL_ERR_INVALID_PARAM;
    }

    notify_msg_t notify_msg;
    memset(&notify_msg, 0, sizeof(notify_msg));
    strncpy(notify_msg.name, local_agent_name_.c_str(), sizeof(notify_msg.name) - 1);
    memcpy(notify_msg.msg, serialized.c_str(), serialized.size());
""",
        """    notify_msg_t notify_msg;
    notify_msg.name = local_agent_name_;
    notify_msg.msg = serialized;
""",
    ),
)


def patch(path: Path) -> str:
    text = path.read_text()
    if ALREADY_PATCHED in text and "sizeof(notify_msg_t::msg)" not in text:
        return "already-patched"

    updated = text
    missing = []
    for old, new in REPLACEMENTS:
        if old not in updated:
            missing.append(old[:60].replace("\n", " "))
            continue
        updated = updated.replace(old, new, 1)

    if missing:
        raise SystemExit(
            "NIXL uccl_backend.cpp notify_msg patterns changed; update "
            f"{Path(__file__).name}. Missing: {missing}"
        )

    path.write_text(updated)
    return "patched"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-uccl_backend.cpp>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    print(f"{patch(path)} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
