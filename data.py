import sqlite3
import pandas as pd # Pandas
import os
from datetime import datetime

import numpy as np
from sklearn.linear_model import LinearRegression # Scikit-learn

class Database:
    def __init__(
            self,
            data: pd.DataFrame = pd.DataFrame(),
            name: str = 'database',
            backup_directory: str = 'backups'
    ):
        self.data = data
        self.name = name
        self.backup_directory = backup_directory

        self.path = f'{self.name}.db'
        self.id_column = 'id'

        self._process()

    def __str__(self) -> str:
        return f'Database(\n\tpath = {self.path}, \n\tcolumns = {self.data.columns.to_list()}, \n\tregisters = {len(self.data)}\n)'
    
    def __repr__(self) -> str:
        rep = f'Database(path = {self.path}, columns = {self.data.columns.to_list()}, registers = {len(self.data)}):'

        for row in self.data.itertuples(index = False):
            rep += f'\n\t{row}'

        return rep

    #|--------------------------------------------------------------|
    #| GETTERS AND SETTERS:                                         |
    #|--------------------------------------------------------------|
    def get_data(self) -> pd.DataFrame:
        '''
        Retorna o DataFrame completo.
        '''
        return self.data
    
    def get_data_by_id(self, id: int) -> pd.DataFrame:
        '''
        Retorna os dados do DataFrame com base no id fornecido.
        '''
        return self.data[self.data[self.id_column] == id]
    
    def get_name(self) -> str:
        '''
        Retorna o nome da tabela do banco de dados.
        '''
        return self.name
    
    def get_backup_directory(self) -> str:
        '''
        Retorna o diretório de backup.
        '''
        return self.backup_directory
    
    def get_backup_paths(self) -> list[str]:
        '''
        Retorna uma lista com os caminhos dos arquivos de backup existentes no diretório de backup.
        '''
        if not os.path.exists(self.backup_directory):
            return []
        
        backup_files = [
            f for f in os.listdir(self.backup_directory)
            if os.path.isfile(os.path.join(self.backup_directory, f)) and f.startswith(self.name) and f.endswith('.db')
        ]
        backup_paths = [os.path.join(self.backup_directory, f) for f in backup_files]
        
        return backup_paths
    
    def get_last_backup_date(self) -> datetime | None:
        '''
        Retorna a data do último backup realizado.
        '''
        backup_paths = self.get_backup_paths()
        
        if not backup_paths:
            return None
        
        latest_backup = max(backup_paths, key=os.path.getctime)
        last_backup_date = datetime.fromtimestamp(os.path.getctime(latest_backup))
        
        return last_backup_date
    
    def get_path(self) -> str:
        '''
        Retorna o caminho do banco de dados SQLite.
        '''
        return self.path
    
    #|--------------------------------------------------------------|
    #| Summary methods:
    def get_size(self) -> int:
        '''
        Retorna o número de registros no DataFrame.
        '''
        return len(self.data)
    
    def get_sum(self, column_names: str | list[str]) -> float | list[float]:
        '''
        Retorna a soma dos valores das colunas passadas do DataFrame.
        '''
        if isinstance(column_names, list):
            return [self.data[column_name].sum() for column_name in column_names]
        
        return self.data[column_names].sum()
    
    def get_mean(self, column_names: str | list[str]) -> float | list[float]:
        '''
        Retorna a média dos valores das colunas passadas do DataFrame.
        '''
        if isinstance(column_names, list):
            return [self.data[column_name].mean() for column_name in column_names]
        
        return self.data[column_names].mean()
    
    def get_frequency(self, column_name: str) -> pd.DataFrame:
        '''
        Retorna a frequência dos valores únicos de uma coluna do DataFrame.
        '''
        frequency = self.data[column_name].value_counts().reset_index()
        frequency.columns = [column_name, 'frequency']
        
        return frequency
    
    def get_variance(self, column_names: str | list[str]) -> float | list[float]:
        '''
        Retorna a variância dos valores das colunas passadas do DataFrame.
        '''
        if isinstance(column_names, list):
            return [self.data[column_name].var() for column_name in column_names]
        
        return self.data[column_names].var()
    
    def get_correlation(self, column_name1: str, column_name2: str) -> float:
        '''
        Retorna a correlação entre duas colunas do DataFrame.
        '''
        return self.data[column_name1].corr(self.data[column_name2])
    
    def get_standard_deviation(self, column_names: str | list[str]) -> float | list[float]:
        '''
        Retorna o desvio padrão dos valores das colunas passadas do DataFrame.
        '''
        if isinstance(column_names, list):
            return [self.data[column_name].std() for column_name in column_names]
        
        return self.data[column_names].std()
    
    def get_predicted_values(self, predict_column: str | list[str], periods: int = 1) -> pd.DataFrame:
        '''
        Analisa a corelação e retorna um DataFrame com a previsão dos valores futuros baseado em regressão linear.
        '''
        if not isinstance(predict_column, list):
            predict_column = [predict_column]

        df_predicted = pd.DataFrame()
        
        x = self.data.index.values.reshape(-1, 1)

        for column in predict_column:
            y = self.data[column].values

            model = LinearRegression()
            model.fit(x, y)

            last_index = self.data.index[-1]
            future_indices = np.arange(last_index + 1, last_index + periods + 1).reshape(-1, 1)
            predicted_values = model.predict(future_indices)

            df_predicted = pd.concat([
                df_predicted,
                pd.DataFrame({
                    column: predicted_values
                })
            ])
        
        df_predicted.set_index(future_indices.flatten(), inplace=True)
        
        return df_predicted

        '''x = self.data.index.values.reshape(-1, 1)
        y = self.data[predict_column].values
        
        model = LinearRegression()
        model.fit(x, y)
        
        last_index = self.data.index[-1]
        future_indices = np.arange(last_index + 1, last_index + periods + 1).reshape(-1, 1)
        predicted_values = model.predict(future_indices)
        
        df_predicted = pd.DataFrame({
            predict_column: predicted_values
        })
        
        df_predicted.set_index(future_indices.flatten(), inplace=True)
        
        return df_predicted'''
    
    def get_error_rate(self, actual_column: str, predicted_column: str) -> float:
        '''
        Retorna a taxa de erro entre os valores reais e previstos de duas colunas do DataFrame.
        '''
        total = len(self.data)
        incorrect = (self.data[actual_column] != self.data[predicted_column]).sum()
        
        return incorrect / total if total > 0 else 0.0

    #|--------------------------------------------------------------|
    #| AUXILIARY METHODS:                                           |
    #|--------------------------------------------------------------|
    def _process(self):
        '''
        Processa os dados existentes no atributo data.
        '''
        for column in self.data.columns.to_list():
            if self.data[column].dtype == 'string' or self.data[column].dtype == 'object':
                self.data[column] = self.data[column] \
                    .str.strip() \
                    .str.replace(r'\s+', ' ', regex = True) \
                    .str.upper()
    
    #|--------------------------------------------------------------|
    #| MAIN METHODS:                                                |
    #|--------------------------------------------------------------|
    def save(self):
        '''
        Salva o DataFrame pandas no banco de dados SQLite.
        '''
        if not isinstance(self.data, pd.DataFrame) or self.data.empty:
            print(f'❌ Erro: {self.data} não é um DataFrame pandas válido ou está vazio.')

            return

        try:
            conn = sqlite3.connect(self.path)

            self.data.to_sql(
                name = self.name,
                con = conn,
                if_exists = 'replace',
                index = False
            )

            conn.close()

            print(f"✅ DataFrame salvo com sucesso na tabela '{self.name}'.")
        except sqlite3.Error as e:
            print(f"❌ Erro ao salvar o DataFrame no SQLite: {e}")
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")

    def load(self):
        '''
        Carrega os dados do banco de dados SQLite diretamente para o atributo data como um DataFrame pandas.
        '''
        try:
            with sqlite3.connect(self.path) as conn:
                query = f"SELECT * FROM {self.name}"
                self.data = pd.read_sql(query, conn)

                self._process()

                print(f"✅ Dados carregados: {len(self.data)} registros.")
                
        except sqlite3.Error as e:
            self.data = pd.DataFrame()

            print(f"❌ Erro ao carregar dados: {e}")

    def backup(self):
        '''
        Realiza o backup do banco de dados SQLite no local especificado.
        '''
        if not os.path.exists(self.backup_directory):
            try:
                os.makedirs(self.backup_directory)
            except OSError as e:
                print(f"❌ Erro ao criar diretório de backup: {e}")

                return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        db_filename = os.path.basename(self.path) 
        backup_filename = db_filename.replace('.db', f'_backup_{timestamp}.db')
        backup_path = os.path.join(self.backup_directory, backup_filename)
        
        try:
            with sqlite3.connect(self.path) as source_conn:
                with sqlite3.connect(backup_path) as destination_conn:
                    source_conn.backup(destination_conn)
            
            print(f"✅ Backup concluído com sucesso! Salvo em: {backup_path}")
            
        except sqlite3.Error as e:
            print(f"❌ Erro durante o backup: {e}")
            
        except Exception as e:
            print(f"❌ Erro inesperado durante o backup: {e}")

    def sort(self, column_names: str | list[str], ascending: bool | list[bool] = True):
        '''
        Ordena o DataFrame pelos nomes das colunas especificadas.
        '''
        self.data = self.data.sort_values(by = column_names, ascending = ascending)

    #|--------------------------------------------------------------|
    #| CRUD methods:
    def add(self, new_data: pd.DataFrame | dict):
        '''
        Adiciona novos dados ao DataFrame existente.
        '''
        if isinstance(new_data, dict):
            new_data = pd.DataFrame(new_data)
        
        for _, row in new_data.iterrows():
            new_row = pd.DataFrame([row])
            self.data = pd.concat([self.data, new_row], ignore_index=True)
            self.data = self.data.drop_duplicates(subset=[self.id_column], keep='last')

        self._process()
    
    def remove(self, data: pd.DataFrame | dict):
        '''
        Remove dados do DataFrame existente.
        '''
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        ids_to_remove = data[self.id_column].tolist()
        self.data = self.data[~self.data[self.id_column].isin(ids_to_remove)]

    #|--------------------------------------------------------------|
    #| Filter methods:
    def filter_by_interval(self, column_name: str, start_value = None, end_value = None):
        '''
        Filtra o DataFrame com base em um intervalo de valores de uma coluna específica.
        '''
        if self.data[column_name].dtype == 'string' or self.data[column_name].dtype == 'object':
            if start_value is not None:
                start_value = str(start_value).strip().upper()
            if end_value is not None:
                end_value = str(end_value).strip().upper()

        if start_value is not None and end_value is not None:
            self.data = self.data[(self.data[column_name] >= start_value) & (self.data[column_name] <= end_value)]
        elif start_value is not None:
            self.data = self.data[self.data[column_name] >= start_value]
        elif end_value is not None:
            self.data = self.data[self.data[column_name] <= end_value]

    def filter_by_values(self, column_name: str, values):
        '''
        Filtra o DataFrame com base em um valor específico de uma coluna.
        '''
        if not isinstance(values, list):
            values = [values]

        if self.data[column_name].dtype == 'string' or self.data[column_name].dtype == 'object':
            values = [str(value).strip().upper() for value in values]

        self.data = self.data[self.data[column_name].isin(values)]

    #|--------------------------------------------------------------|
    #| Export methods:
    def to_dict(self):
        pass

    def to_json(self):
        pass

    def to_csv(self):
        pass

    def to_excel(self):
        pass

    #|--------------------------------------------------------------|
    #| Import methods:
    def from_dict(self, data_dict):
        pass

    def from_json(self, json_str):
        pass

    def from_csv(self, csv_path):
        pass

    def from_excel(self, excel_path):
        pass

#|--------------------------------------------------------------|
#| TESTS:                                                       |
#|--------------------------------------------------------------|
if __name__ == "__main__":
    data = {
        'id': [1, 2, 3, 4, 5],
        'Nome': ['Alice', 'Bob', 'João', 'Leandro', 'Leandro'],
        'Data': [datetime(2025, 11, 26), datetime(2025, 5, 27), datetime(2025, 5, 28), datetime(2024, 11, 29), datetime(2025, 11, 30)],
        'Idade': [20, 25, 30, 36, 36],
        'Saldo': [320.00, 450.50, 1600.75, 2800.00, 2800.00],
    }
    dataframe = pd.DataFrame(data)
    database = Database(dataframe)

    database.save()

    print(database.get_predicted_values(['Saldo', 'Idade'], 3))
    print(database.get_data())

    print(database.__repr__())

    #database.backup()
