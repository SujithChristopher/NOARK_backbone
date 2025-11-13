import msgpack
import numpy as np

def decode_numpy(obj):
    """Decode msgpack numpy format"""
    if isinstance(obj, dict) and b'nd' in obj:
        dtype = np.dtype(obj[b'kind'].decode() + str(obj[b'type']))
        return np.frombuffer(obj[b'data'], dtype=dtype).reshape(obj[b'shape'])
    return obj

# Load and examine the chessboard data - handle multiple objects
data_list = []
with open('newdata/chessboard_20251113_153002.msgpack', 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False)
    for item in unpacker:
        decoded = decode_numpy(item)
        data_list.append(decoded)

print(f"Total number of frames: {len(data_list)}")
print("\nFirst frame structure:")
if len(data_list) > 0:
    corners = data_list[0]
    print(f"Type: {type(corners)}")
    print(f"Shape: {corners.shape}")
    print(f"Dtype: {corners.dtype}")
    print(f"\nFirst 3 corners:")
    print(corners[:3])

    # Check spread of corners
    print(f"\nCorner statistics:")
    print(f"  X range: {corners[:, 0, 0].min():.2f} to {corners[:, 0, 0].max():.2f}")
    print(f"  Y range: {corners[:, 0, 1].min():.2f} to {corners[:, 0, 1].max():.2f}")
