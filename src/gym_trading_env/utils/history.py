import numpy as np

class History:
    def __init__(self, max_size = 10000):
        self.height = max_size
    def set(self, **kwargs):
        # Flattening the inputs to put it in np.array
        self.columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                self.columns.extend([f"{name}_{i}" for i in range(len(value))])
            elif isinstance(value, dict):
                self.columns.extend([f"{name}_{key}" for key in value.keys()])
            else:
                self.columns.append(name)
        
        self.column_map = {column: i for i, column in enumerate(self.columns)}
        self.width = len(self.columns)
        self.history_storage = np.zeros(shape=(self.height, self.width), dtype= 'O')
        self.size = 0
        self.add(**kwargs)
    def add(self, **kwargs):
        values = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                values.extend(value[:])
            elif isinstance(value, dict):
                values.extend(list(value.values()))
            else:
                values.append(value)

        if len(values) == self.width:
            self.history_storage[self.size, :] = values
            self.size = min(self.size+1, self.height)
        else:
            raise ValueError(f"Make sure that your inputs match the initial ones... Expected width : {self.width}. New width {len(values)}")
    def __len__(self):
        return self.size
    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            column, t = arg
            column_index = self.column_map.get(column)
            if column_index is None:
                raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][t, column_index]
        if isinstance(arg, int):
            t = arg
            return dict(zip(self.columns, self.history_storage[:self.size][t]))
        if isinstance(arg, str):
            column = arg
            column_index = self.column_map.get(column)
            if column_index is None:
                raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][:, column_index]
        if isinstance(arg, list):
            columns = arg
            column_indexes = []
            for column in columns:
                column_index = self.column_map.get(column)
                if column_index is None:
                    raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
                column_indexes.append(column_index)
            return self.history_storage[:self.size][:, column_indexes]

    def __setitem__(self, arg, value):
        column, t = arg
        try:
            column_index = self.columns.index(column)
        except ValueError:
            raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
        self.history_storage[:self.size][t, column_index] = value