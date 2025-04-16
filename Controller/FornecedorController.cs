using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace FornecedorIA
{
    public class FornecedorData
    {
        [LoadColumn(0)] public float TempoEntrega { get; set; }
        [LoadColumn(1)] public float Qualidade { get; set; }
        [LoadColumn(2)] public float Custo { get; set; }
        [LoadColumn(3)] public string? Categoria { get; set; }
    }

    public class FornecedorPrediction
    {
        [ColumnName("PredictedLabel")] public string CategoriaPredita { get; set; } = default!;
    }

    public class MetricasClasse
    {
        public string Classe { get; set; } = string.Empty;
        public double Precisao { get; set; }
        public double Revocacao { get; set; }
        public double F1Score { get; set; }
    }

    public class UploadCsvRequest
    {
        public IFormFile Arquivo { get; set; } = default!;
    }

    public class ModelTrainer
    {
        private readonly MLContext _mlContext = new();
        private ITransformer _modelo = default!;
        private PredictionEngine<FornecedorData, FornecedorPrediction> _predEngine = default!;

        public void TreinarModelo(string caminhoCsv)
        {
            var dados = _mlContext.Data.LoadFromTextFile<FornecedorData>(
                path: caminhoCsv,
                separatorChar: ',',
                hasHeader: true);

            var pipelineDeAprendizado = _mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(FornecedorData.Categoria))
                .Append(_mlContext.Transforms.Concatenate("Features", nameof(FornecedorData.TempoEntrega), nameof(FornecedorData.Qualidade), nameof(FornecedorData.Custo)))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _modelo = pipelineDeAprendizado.Fit(dados);

            _predEngine = _mlContext.Model.CreatePredictionEngine<FornecedorData, FornecedorPrediction>(_modelo);
        }

        public string Classificar(FornecedorData dados)
        {
            var predicao = _predEngine.Predict(dados);
            return predicao.CategoriaPredita;
        }

        public MulticlassClassificationMetrics AvaliarModelo(string caminhoTesteCsv)
        {
            var dadosTeste = _mlContext.Data.LoadFromTextFile<FornecedorData>(
                path: caminhoTesteCsv,
                separatorChar: ',',
                hasHeader: true);

            var dadosTransformados = _modelo.Transform(dadosTeste);
            var metrics = _mlContext.MulticlassClassification.Evaluate(dadosTransformados);
            return metrics;
        }

        public List<MetricasClasse> CalcularMetricasPorClasse(MulticlassClassificationMetrics metrics)
        {
            var matriz = metrics.ConfusionMatrix.Counts;
            var resultado = new List<MetricasClasse>();

            for (int i = 0; i < matriz.Count; i++)
            {
                double tp = matriz[i][i];
                double fn = 0, fp = 0;

                for (int j = 0; j < matriz.Count; j++)
                {
                    if (i != j)
                    {
                        fn += matriz[i][j];
                        fp += matriz[j][i];
                    }
                }

                double precisao = tp + fp == 0 ? 0 : tp / (tp + fp);
                double revocacao = tp + fn == 0 ? 0 : tp / (tp + fn);
                double f1 = precisao + revocacao == 0 ? 0 : 2 * (precisao * revocacao) / (precisao + revocacao);

                resultado.Add(new MetricasClasse
                {
                    Classe = $"Classe {i}",
                    Precisao = Math.Round(precisao, 3),
                    Revocacao = Math.Round(revocacao, 3),
                    F1Score = Math.Round(f1, 3)
                });
            }

            return resultado;
        }
    }

    [ApiController]
    [Route("api/[controller]")]
    public class FornecedorController : ControllerBase
    {
        private static readonly ModelTrainer _trainer = new();
        private static bool _modeloTreinado = false;

        [HttpPost("upload-csv")]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> UploadCsv([FromForm] UploadCsvRequest request)
        {
            var arquivo = request.Arquivo;
            if (arquivo == null || arquivo.Length == 0)
                return BadRequest("Arquivo CSV não enviado.");

            var caminho = Path.Combine(Path.GetTempPath(), arquivo.FileName);
            using (var stream = new FileStream(caminho, FileMode.Create))
            {
                await arquivo.CopyToAsync(stream);
            }

            _trainer.TreinarModelo(caminho);
            _modeloTreinado = true;
            return Ok("Modelo treinado com sucesso a partir do arquivo enviado.");
        }

        [HttpPost("carregar-modelo")]
        public IActionResult CarregarModelo([FromQuery] string caminhoCsv)
        {
            _trainer.TreinarModelo(caminhoCsv);
            _modeloTreinado = true;
            return Ok("Modelo treinado com sucesso.");
        }

        [HttpPost("classificar")]
        public ActionResult<string> Classificar([FromBody] FornecedorData dados)
        {
            if (!_modeloTreinado)
                return BadRequest("Modelo ainda não foi treinado.");

            var resultado = _trainer.Classificar(dados);
            return Ok(resultado);
        }

        [HttpPost("avaliar-modelo")]
        public IActionResult AvaliarModelo([FromQuery] string caminhoTesteCsv)
        {
            if (!_modeloTreinado)
                return BadRequest("Modelo ainda não foi treinado.");

            var metrics = _trainer.AvaliarModelo(caminhoTesteCsv);
            var metricasPorClasse = _trainer.CalcularMetricasPorClasse(metrics);

            return Ok(new
            {
                metrics.MacroAccuracy,
                metrics.MicroAccuracy,
                metrics.LogLoss,
                metrics.LogLossReduction,
                MatrizConfusao = metrics.ConfusionMatrix.Counts,
                MetricasPorClasse = metricasPorClasse
            });
        }
    }
}
