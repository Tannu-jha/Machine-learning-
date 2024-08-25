#data processing of dataset
def process_data(file_path, numerical_columns):  
    try: 
        df = pd.read_csv(file_path)
        
        print("Initial dataset preview:")
        print(df.head())
        
        print("Dataset info:")
        print(df.info())
        
        for col in numerical_columns:
            if df[col].isnull().any():
                print(f"Column '{col}' has missing values.")
                df[col].fillna(df[col].mean(), inplace=True)  
                
        for col in numerical_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the dataset.")
                
        if df[numerical_columns].shape[0] == 0:
            raise ValueError("No data available after handling missing values.")
            
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        
        print("Data standardized. Example of processed data:")
        return df

    except FileNotFoundError:
        print("Error: The file was not found. Please check the file path.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def run_data_processing():
    file_path = r"C:\Users\tannu\Desktop\people.csv"
    numerical_columns_input = input("Enter numerical columns to standardize, separated by commas (e.g., Age, Salary): ")
    numerical_columns = [col.strip() for col in numerical_columns_input.split(',')]
    processed_df = process_data(file_path, numerical_columns)
    if processed_df is not None and not processed_df.empty:
        display(processed_df.head())
    else:
        print("Processed DataFrame is empty.")
run_data_processing()
