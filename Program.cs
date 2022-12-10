/*This tutorial illustrates how to build a regression model using ML.NET to predict prices, specifically, New York City taxi fares.
 * 
 * 
 * The label is the column you want to predict. The identified Featuresare the inputs you give the model to predict the Label.
 * 
 * 
 * The provided data set contains the following columns:
 *   vendor_id: The ID of the taxi vendor is a feature.
 *   rate_code: The rate type of the taxi trip is a feature.
 *   passenger_count: The number of passengers on the trip is a feature.
 *   trip_time_in_secs: The amount of time the trip took. You want to predict the fare of the trip before the trip is completed. At that moment, you don't know how long the trip would take. Thus, the trip time is not a feature and you'll exclude this column from the model.
 *   trip_distance: The distance of the trip is a feature.
 *   payment_type: The payment method (cash or credit card) is a feature.
 *   fare_amount: The total taxi fare paid is the label.
 *   
 *   
 *   
 *   
 *   

*/



using Microsoft.ML;
using TaxiFarePrediction;

//_trainDataPath contains the path to the file with the data set used to train the model.
string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");

//_testDataPath contains the path to the file with the data set used to evaluate the model.
string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");

//_modelPath contains the path to the file where the trained model is stored.
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

/*All ML.NET operations start in the MLContext class. 
 * Initializing mlContext creates a new ML.NET environment that can be shared across the model creation workflow objects.
 * It's similar, conceptually, to DBContext in Entity Framework.*/
MLContext mlContext = new MLContext(seed: 0);

/*The Train() method executes the following tasks:
    * Loads the data.
    * Extracts and transforms the data.
    * Trains the model.
    * Returns the model.
*/
var model = Train(mlContext, _trainDataPath);


/*
 * The Evaluate method executes the following tasks:
    * Loads the test dataset.
    * Creates the regression evaluator.
    * Evaluates the model and creates metrics.
    * Displays the metrics.
*/
Evaluate(mlContext, model);

/*
 * The TestSinglePrediction method executes the following tasks:
    * Creates a single comment of test data.
    * Predicts fare amount based on test data.
    * Combines test data and predictions for reporting.
    * Displays the predicted results.
*/
TestSinglePrediction(mlContext, model);




ITransformer Train(MLContext mlContext, string dataPath)
{
    /*ML.NET uses the IDataView interface as a flexible, efficient way of describing numeric or text tabular data. 
     * IDataView can load either text files or in real time (for example, SQL database or log files). 
     * */
IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

    /*As you want to predict the taxi trip fare, the FareAmount column is the Label that you will predict (the output of the model).
     * Use the CopyColumnsEstimator transformation class to copy FareAmount.
     * */
    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
       /* The algorithm that trains the model requires numeric features, 
        * so you have to transform the categorical data(VendorId, RateCode, and PaymentType) values 
        * into numbers(VendorIdEncoded, RateCodeEncoded, and PaymentTypeEncoded).
        * To do that, use the OneHotEncodingTransformer transformation class, 
        * which assigns different numeric key values to the different values in each of the columns*/
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
         /*The last step in data preparation combines all of the feature columns into the Features column using the mlContext.Transforms.
             * Concatenate transformation class. By default, a learning algorithm processes only features from the Features column. 
         */
    .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
        /*This problem is about predicting a taxi trip fare in New York City. 
         * At first glance, it may seem to depend simply on the distance traveled. 
         * However, taxi vendors in New York charge varying amounts for other factors such as additional passengers
         * or paying with a credit card instead of cash.
         * You want to predict the price value, which is a real value, based on the other factors in the dataset. 
         * To do that, you choose a regression machine learning task.
         * 
         * Append the FastTreeRegressionTrainer machine learning task to the data transformation definitions
         */
    .Append(mlContext.Regression.Trainers.FastTree());

    //Fit the model to the training dataview and return the trained model 
    var model = pipeline.Fit(dataView);
    /*The Fit() method trains your model by transforming the dataset and applying the training.
     * Return the trained model
     */
    return model;

}

//Next, evaluate your model performance with your test data for quality assurance and validation.
void Evaluate(MLContext mlContext, ITransformer model)
{
    /*Load the test dataset using the LoadFromTextFile() method. 
     * Evaluate the model using this dataset as a quality check
     */
    IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

    /* Next, transform the Test data by adding the following code to Evaluate():
     * The Transform() method makes predictions for the test dataset input rows.
     */
    var predictions = model.Transform(dataView);

    /* The RegressionContext.Evaluate method computes the quality metrics for the PredictionModel using the specified dataset.
     * It returns a RegressionMetrics object that contains the overall metrics computed by regression evaluators.
     * To display these to determine the quality of the model, you need to get the metrics first.
     * 
     * Once you have the prediction set, the Evaluate() method assesses the model, which compares the predicted values
     * with the actual Labels in the test dataset and returns metrics on how the model is performing.
     */
    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");

    /*RSquared is another evaluation metric of the regression models.
     * RSquared takes values between 0 and 1. The closer its value is to 1, the better the model is.
     */
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");

    //RMS is one of the evaluation metrics of the regression model. The lower it is, the better the model is.
    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");

}


void TestSinglePrediction(MLContext mlContext, ITransformer model)
{
    //Use the PredictionEngine to predict the fare
    var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

    /*
       The PredictionEngine is a convenience API, which allows you to perform a prediction on a single instance of data.
       PredictionEngine is not thread - safe.It's acceptable to use in single-threaded or prototype environments. 
       For improved performance and thread safety in production environments, use the PredictionEnginePool service, 
       which creates an ObjectPool of PredictionEngine objects for use throughout your application. 
       See this guide on how to use PredictionEnginePool in an ASP.NET Core Web API.
   */
    var taxiTripSample = new TaxiTrip()
    {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripTime = 1140,
        TripDistance = 3.75f,
        PaymentType = "CRD",
        FareAmount = 0 // To predict. Actual/Observed = 15.5
    };

    /*  Next, predict the fare based on a single instance of the taxi trip data and pass it to the PredictionEngine
        The Predict() function makes a prediction on a single instance of data.
    */
    var prediction = predictionFunction.Predict(taxiTripSample);

   

    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
    Console.WriteLine($"**********************************************************************");
}