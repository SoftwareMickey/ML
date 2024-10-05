# * Create a function that will be used to convert data to be predicted from objects to
# Todo -> Function to reshape data
def reshaped_data(data, np):

  # Reshape data from 9 features to 13 features
  desired_shape = 13

  # check the current features
  current_features = len(data[0])

  # calculate number of dummy features needed
  if current_features < desired_shape:
      padding = [0.0] * (desired_shape - current_features)

      # Add the padding to the start of the array
      reshaped_data = np.array(padding + data[0]).reshape(1, -1)
  else:
      reshaped_data = np.array(data).reshape(1, -1)

  print(f"Reshaped data: {data}")
  return reshaped_data


#  * Create a function that will be used to convert data to be predicted from objects to numbers
def convert_to_numbers(data):
    converted = []
    for item in data:
        row = []
        for value in item:
            try:
                # Attempt to convert to float first
                row.append(float(value))
            except ValueError:
                # If float conversion fails, keep as string
                row.append(value)
        converted.append(row)
    return converted

# Todo: Create a function that will be used to convert data to be predicted from objects to numbers
def accept_and_convert_data(input_data):
  prediction_data = list(input_data.values());

  # print the prediction data
  print("Before conversion of data: " + str(prediction_data))

  # Convert all column that have string columns
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder

  prediction_data = np.array(prediction_data).reshape(1, -1)

  ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [8])], remainder = 'passthrough')
  ct.fit(prediction_data)
  prediction_data = ct.transform(prediction_data)

  # Change the prediction data from string to numbers
  prediction_data = convert_to_numbers(prediction_data)

  # print the prediction data after transformation
  print("After data conversion of data: " + str(prediction_data))

  # reshaped_data = np.array(prediction_data).reshape(13, -1)
  # print("Reshaped data: " + str(reshaped_data))

  return prediction_data


# Todo: Define methods reshape the data...we
def load_data_and_predict(data, prediction_model):
  prediction = prediction_model.predict(data)
  return prediction