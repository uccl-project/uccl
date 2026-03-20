import torch

send = torch.load('warmup_send_client_1073741824.pt', weights_only=True)
recv = torch.load('warmup_recv_server_1073741824.pt', weights_only=True)

print(f'Send shape: {send.shape}, dtype: {send.dtype}')
print(f'Recv shape: {recv.shape}, dtype: {recv.dtype}')
print(f'Send first 20: {send[:20]}')
print(f'Recv first 20: {recv[:20]}')
print(f'Exact match: {torch.equal(send, recv)}')

if not torch.equal(send, recv):
    diff_mask = send != recv
    num_diff = diff_mask.sum().item()
    total = send.numel()
    print(f'Mismatched elements: {num_diff}/{total} ({num_diff/total*100:.6f}%)')
    diff_indices = torch.where(diff_mask)[0]
    for idx in diff_indices[:10]:
        print(f'  idx {idx.item()}: send={send[idx].item()}, recv={recv[idx].item()}')
    if num_diff > 10:
        print('  ...')
        for idx in diff_indices[-5:]:
            print(f'  idx {idx.item()}: send={send[idx].item()}, recv={recv[idx].item()}')
