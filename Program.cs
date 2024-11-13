using Microsoft.ML;
using GitHubIssueClassification;
//using System.Data;
//using System.IO;

string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

MLContext _mlContext;
PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
ITransformer _trainedModel;
IDataView _trainingDataView;

_mlContext = new MLContext(seed: 0);
_trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

//Extract features & Transform the data method
var pipeline = ProcessData();

//Extract features & Transform the data
IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
        .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
        .AppendCacheCheckpoint(_mlContext);

    return pipeline; //Returns the processing pipeline
}


//Build and train the model
var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

//Build and train the model method
IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    //Trian the model
    _trainedModel = trainingPipeline.Fit(trainingDataView);
    _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

    //Predict with the trained model
    GitHubIssue issue = new GitHubIssue()
    {
        Title = "WebSockets communication is slow in my machine",
        Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    };

    var prediction = _predEngine.Predict(issue);

    Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

    return trainingPipeline; //Returns the training pipeline
}

Evaluate(_trainingDataView.Schema);

//Evaluate the model
void Evaluate(DataViewSchema trainingDataViewSchema)
{
    var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);
    var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");

    SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
}

//Save the model as a file method
void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
    //PredictIssue();
}

PredictIssue();
//Predict Issues method
void PredictIssue()
{
    ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
    GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };
    _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
    var prediction = _predEngine.Predict(singleIssue);
    Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
}