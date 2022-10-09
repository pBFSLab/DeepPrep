class Source:

    def __init__(self, CPU_n: int = 1, GPU_MB: int = 0, RAM_MB: int = 0, IO_write_MB: int = 0, IO_read_MB: int = 0):
        self.source = {
            'CPU_n': CPU_n,
            'GPU_MB': GPU_MB,
            'RAM_MB': RAM_MB,
            'IO_write_MB': IO_write_MB,
            'IO_read_MB': IO_read_MB,
        }

    def __add__(self, other):
        add_result = {}
        for i, j in other.source.items():
            add_result[i] = self.source[i] + other.source[i]

        return Source(**add_result)

    def __sub__(self, other):
        sub_result = {}
        for i, j in other.source.items():
            sub_result[i] = self.source[i] - other.source[i]
        return Source(**sub_result)

    def __iter__(self):
        return iter(self.source.values())
