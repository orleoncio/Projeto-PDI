import { useState } from 'react'
import {
  Activity,
  Layers,
  Cpu,
  FileText,
  Sliders,
  BarChart3,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  Brain,
  Zap,
  Info
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts'

// Dados comparativos oficiais vs replicação (Tabela 3 do Artigo)
const metricasVGG16 = [
  { metrica: 'Accuracy', artigoSGD: 98.16, nossoSGD: 97.98, artigoAdam: 19.96, nossoAdam: 21.73 },
  { metrica: 'F1-Score', artigoSGD: 98.05, nossoSGD: 97.64, artigoAdam: 5.55, nossoAdam: 5.95 },
  { metrica: 'Precision', artigoSGD: 98.14, nossoSGD: 97.56, artigoAdam: 3.33, nossoAdam: 3.62 },
  { metrica: 'Recall', artigoSGD: 97.98, nossoSGD: 97.73, artigoAdam: 16.67, nossoAdam: 16.67 },
]

const radarData = [
  { metric: 'Acurácia', SGD: 97.98, Adam: 21.73 },
  { metric: 'F1-Score', SGD: 97.64, Adam: 5.95 },
  { metric: 'Precisão', SGD: 97.56, Adam: 3.62 },
  { metric: 'Sensibilidade', SGD: 97.73, Adam: 16.67 },
]

const classesCelulares = [
  { nome: 'Homogêneo', padrao: 'Fluorescência difusa no núcleo intermitente', celulas: '~22%' },
  { nome: 'Pontilhado Fino', padrao: 'Grânulos diminutos por todo o nucleoplasma', celulas: '~18%' },
  { nome: 'Pontilhado Grosso', padrao: 'Grânulos densos e proeminentes', celulas: '~16%' },
  { nome: 'Centrômero', padrao: 'Pontos discretos alinhados nos cromossomos', celulas: '~15%' },
  { nome: 'Nucleolar', padrao: 'Marcação concentrada nos nucléolos', celulas: '~14%' },
  { nome: 'Citoplasmático', padrao: 'Fluorescência no citoplasma preservado', celulas: '~15%' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState<'visao-geral' | 'pre-processamento' | 'vgg16' | 'squeezenet'>('visao-geral')
  const [optimizerView, setOptimizerView] = useState<'sgd' | 'adam' | 'ambos'>('ambos')

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400">
              <Brain className="w-6 h-6" />
            </div>
            <div>
              <span className="font-semibold text-lg tracking-tight text-white block leading-tight">
                PDI & Deep Learning · HEp-2 Diagnosis
              </span>
              <span className="text-xs text-slate-400">
                Apresentação Interativa de Resultados
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              Branch: feat/apresentacao
            </span>
          </div>
        </div>
      </header>

      {/* Hero Banner */}
      <section className="bg-gradient-to-b from-slate-900 to-slate-950 border-b border-slate-800/80 py-10 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-medium mb-3">
            <FileText className="w-3.5 h-3.5" />
            Artigo: Preprocessing and Optimizer Impact on Image-Based Autoimmune Disease Diagnosis
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white max-w-4xl">
            Impacto do Pré-processamento e Otimizadores na Detecção de Doenças Autoimunes
          </h1>
          <p className="mt-3 text-slate-400 max-w-3xl text-sm sm:text-base leading-relaxed">
            Replicação experimental e análise empírica dos efeitos de algoritmos de otimização (SGD com momentum vs. Adam) e técnicas de pré-processamento de imagens no diagnóstico celular automatizado por fluorescência indireta (HEp-2).
          </p>

          {/* Navigation Tabs */}
          <div className="flex flex-wrap gap-2 mt-8 border-b border-slate-800 pb-px">
            <button
              onClick={() => setActiveTab('visao-geral')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-medium transition-all ${
                activeTab === 'visao-geral'
                  ? 'bg-slate-850 text-emerald-400 border-b-2 border-emerald-400 bg-slate-900'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Info className="w-4 h-4" />
              Visão Geral & Classes
            </button>
            <button
              onClick={() => setActiveTab('pre-processamento')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-medium transition-all ${
                activeTab === 'pre-processamento'
                  ? 'bg-slate-850 text-emerald-400 border-b-2 border-emerald-400 bg-slate-900'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Sliders className="w-4 h-4" />
              Técnicas de PDI
            </button>
            <button
              onClick={() => setActiveTab('vgg16')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-medium transition-all ${
                activeTab === 'vgg16'
                  ? 'bg-slate-850 text-emerald-400 border-b-2 border-emerald-400 bg-slate-900'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              VGG-16 (Confrontação Oficial)
            </button>
            <button
              onClick={() => setActiveTab('squeezenet')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-medium transition-all ${
                activeTab === 'squeezenet'
                  ? 'bg-slate-850 text-emerald-400 border-b-2 border-emerald-400 bg-slate-900'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Zap className="w-4 h-4" />
              SqueezeNet & Eficiência
            </button>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* TAB 1: VISÃO GERAL */}
        {activeTab === 'visao-geral' && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5">
                <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center mb-3">
                  <CheckCircle2 className="w-5 h-5" />
                </div>
                <h3 className="text-white font-semibold mb-1">Replicação SGD</h3>
                <p className="text-sm text-slate-400">
                  Atingimos <strong>97.98% de acurácia</strong> (artigo: 98.16%, delta de apenas -0.18%), validando fielmente a pipeline de data augmentation e fine-tuning.
                </p>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5">
                <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center mb-3">
                  <AlertTriangle className="w-5 h-5" />
                </div>
                <h3 className="text-white font-semibold mb-1">Colapso do Adam Replicado</h3>
                <p className="text-sm text-slate-400">
                  Confirmamos o fenômeno do artigo: o otimizador Adam sofreu colapso metodológico, convergindo para predição de classe única (recall de <strong>16.67% = 1/6</strong>).
                </p>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5">
                <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center mb-3">
                  <Cpu className="w-5 h-5" />
                </div>
                <h3 className="text-white font-semibold mb-1">Arquiteturas Analisadas</h3>
                <p className="text-sm text-slate-400">
                  Comparações entre redes pesadas (<strong>VGG-16</strong> com 537MB) e arquiteturas ultra-leves (<strong>SqueezeNet 1.1</strong> com ~3MB).
                </p>
              </div>
            </div>

            {/* Classes Celulares */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                <Layers className="w-5 h-5 text-emerald-400" />
                Classes Celulares do Dataset HEp-2
              </h2>
              <p className="text-sm text-slate-400 mb-6">
                O conjunto de dados contém imagens de imunofluorescência indireta divididas em 6 classes morfológicas distintas:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {classesCelulares.map((c, idx) => (
                  <div key={c.nome} className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 hover:border-slate-700 transition">
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-medium text-white text-sm">
                        {idx + 1}. {c.nome}
                      </span>
                      <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                        {c.celulas}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {c.padrao}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: PRÉ-PROCESSAMENTO */}
        {activeTab === 'pre-processamento' && (
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                <Sliders className="w-5 h-5 text-emerald-400" />
                Técnicas de PDI e Data Augmentation
              </h2>
              <p className="text-sm text-slate-400 max-w-3xl mb-6">
                O artigo avalia a influência de técnicas de processamento de imagem para realçar os detalhes nucleares das células HEp-2 antes do treinamento das redes convolucionais.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="border border-slate-800 bg-slate-950/80 rounded-xl p-5">
                  <div className="text-xs font-semibold uppercase tracking-wider text-emerald-400 mb-2">Cenário 1</div>
                  <h3 className="font-semibold text-white mb-2">Imagem Original (Sem PDI extra)</h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Mantém os canais originais RGB normalizados com média e desvio padrão do ImageNet.
                  </p>
                  <div className="h-32 bg-slate-900 rounded-lg flex items-center justify-center border border-dashed border-slate-800 text-xs text-slate-500">
                    [Amostra Original HEp-2]
                  </div>
                </div>

                <div className="border border-slate-800 bg-slate-950/80 rounded-xl p-5">
                  <div className="text-xs font-semibold uppercase tracking-wider text-indigo-400 mb-2">Cenário 2</div>
                  <h3 className="font-semibold text-white mb-2">Expansão de Contraste</h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Estiramento linear dos percentis de intensidade para aumentar a separação entre nucleoplasma e grânulos.
                  </p>
                  <div className="h-32 bg-slate-900 rounded-lg flex items-center justify-center border border-dashed border-slate-800 text-xs text-slate-500">
                    [Amostra Contrast Expansion]
                  </div>
                </div>

                <div className="border border-slate-800 bg-slate-950/80 rounded-xl p-5">
                  <div className="text-xs font-semibold uppercase tracking-wider text-purple-400 mb-2">Cenário 3</div>
                  <h3 className="font-semibold text-white mb-2">Data Augmentation Diédrico ($D_4$)</h3>
                  <p className="text-xs text-slate-400 mb-4">
                    8 transformações invariantes: rotações ortogonais (0°, 90°, 180°, 270°) combinadas com reflexões horizontais/verticais.
                  </p>
                  <div className="h-32 bg-slate-900 rounded-lg flex items-center justify-center border border-dashed border-slate-800 text-xs text-slate-500">
                    [Grupo Diédrico D4]
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: VGG-16 CONFRONTAÇÃO */}
        {activeTab === 'vgg16' && (
          <div className="space-y-8">
            {/* Tabela de Confrontação Oficial */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-emerald-400" />
                    Confrontação Oficial: Artigo vs. Nossa Replicação (Tabela 3)
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    VGG-16 sem pesos pré-treinados, 30 épocas, batch size 32.
                  </p>
                </div>
                <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
                  <button
                    onClick={() => setOptimizerView('ambos')}
                    className={`px-3 py-1 text-xs rounded-md transition ${optimizerView === 'ambos' ? 'bg-slate-800 text-white font-medium' : 'text-slate-400'}`}
                  >
                    Ambos
                  </button>
                  <button
                    onClick={() => setOptimizerView('sgd')}
                    className={`px-3 py-1 text-xs rounded-md transition ${optimizerView === 'sgd' ? 'bg-emerald-500/20 text-emerald-300 font-medium' : 'text-slate-400'}`}
                  >
                    SGD
                  </button>
                  <button
                    onClick={() => setOptimizerView('adam')}
                    className={`px-3 py-1 text-xs rounded-md transition ${optimizerView === 'adam' ? 'bg-amber-500/20 text-amber-300 font-medium' : 'text-slate-400'}`}
                  >
                    Adam
                  </button>
                </div>
              </div>

              {/* Tabela */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-950/80 text-xs text-slate-400 uppercase tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-3 px-4">Otimizador</th>
                      <th className="py-3 px-4">Métrica</th>
                      <th className="py-3 px-4">Artigo Oficial</th>
                      <th className="py-3 px-4">Nosso Treino</th>
                      <th className="py-3 px-4">Diferença ($\Delta$)</th>
                      <th className="py-3 px-4">Diagnóstico</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                    {(optimizerView === 'ambos' || optimizerView === 'sgd') && (
                      <>
                        <tr className="hover:bg-slate-800/20">
                          <td className="py-3 px-4 font-bold text-emerald-400" rowSpan={4}>SGD</td>
                          <td className="py-3 px-4 font-sans text-slate-200">Accuracy</td>
                          <td className="py-3 px-4 text-slate-300">98.16%</td>
                          <td className="py-3 px-4 text-emerald-400 font-bold">97.98%</td>
                          <td className="py-3 px-4 text-slate-400">-0.18%</td>
                          <td className="py-3 px-4 font-sans text-slate-300">Replicação fiel de alta precisão</td>
                        </tr>
                        <tr className="hover:bg-slate-800/20">
                          <td className="py-3 px-4 font-sans text-slate-200">F1-Score</td>
                          <td className="py-3 px-4 text-slate-300">98.05%</td>
                          <td className="py-3 px-4 text-emerald-400 font-bold">97.64%</td>
                          <td className="py-3 px-4 text-slate-400">-0.41%</td>
                          <td className="py-3 px-4 font-sans text-slate-300">Excelente balanço entre classes</td>
                        </tr>
                        <tr className="hover:bg-slate-800/20">
                          <td className="py-3 px-4 font-sans text-slate-200">Precision</td>
                          <td className="py-3 px-4 text-slate-300">98.14%</td>
                          <td className="py-3 px-4 text-emerald-400 font-bold">97.56%</td>
                          <td className="py-3 px-4 text-slate-400">-0.58%</td>
                          <td className="py-3 px-4 font-sans text-slate-300">Baixa taxa de falsos positivos</td>
                        </tr>
                        <tr className="hover:bg-slate-800/20">
                          <td className="py-3 px-4 font-sans text-slate-200">Recall</td>
                          <td className="py-3 px-4 text-slate-300">97.98%</td>
                          <td className="py-3 px-4 text-emerald-400 font-bold">97.73%</td>
                          <td className="py-3 px-4 text-slate-400">-0.25%</td>
                          <td className="py-3 px-4 font-sans text-slate-300">Sensibilidade robusta</td>
                        </tr>
                      </>
                    )}

                    {(optimizerView === 'ambos' || optimizerView === 'adam') && (
                      <>
                        <tr className="hover:bg-slate-800/20 border-t border-slate-700/50">
                          <td className="py-3 px-4 font-bold text-amber-400" rowSpan={4}>ADAM</td>
                          <td className="py-3 px-4 font-sans text-slate-200">Accuracy</td>
                          <td className="py-3 px-4 text-slate-300">19.96%</td>
                          <td className="py-3 px-4 text-amber-400 font-bold">21.73%</td>
                          <td className="py-3 px-4 text-slate-400">+1.77%</td>
                          <td className="py-3 px-4 font-sans text-amber-300">Replicação exata do colapso</td>
                        </tr>
                        <tr className="hover:bg-slate-800/20">
                          <td className="py-3 px-4 font-sans text-slate-200">F1-Score</td>
                          <td className="py-3 px-4 text-slate-300">5.55%</td>
                          <td className="py-3 px-4 text-amber-400 font-bold">5.95%</td>
                          <td className="py-3 px-4 text-slate-400">+0.40%</td>
                          <td className="py-3 px-4 font-sans text-slate-300">Preso em mínimo local raso</td>
                        </tr>
                        <tr className="hover:bg-slate-800/20">
                          <td className="py-3 px-4 font-sans text-slate-200">Precision</td>
                          <td className="py-3 px-4 text-slate-300">3.33%</td>
                          <td className="py-3 px-4 text-amber-400 font-bold">3.62%</td>
                          <td className="py-3 px-4 text-slate-400">+0.29%</td>
                          <td className="py-3 px-4 font-sans text-slate-300">Sensível a desbalanceamento local</td>
                        </tr>
                        <tr className="hover:bg-slate-800/20">
                          <td className="py-3 px-4 font-sans text-slate-200">Recall</td>
                          <td className="py-3 px-4 text-slate-300">16.67%</td>
                          <td className="py-3 px-4 text-amber-400 font-bold">16.67%</td>
                          <td className="py-3 px-4 text-slate-400">0.00%</td>
                          <td className="py-3 px-4 font-sans text-amber-300">Predição de classe única (1/6 = 16.67%)</td>
                        </tr>
                      </>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Gráficos Comparativos */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-emerald-400" />
                  Comparativo de Métricas (%) - SGD vs Adam
                </h3>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={metricasVGG16} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="metrica" stroke="#94a3b8" fontSize={12} />
                      <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8 }} />
                      <Legend />
                      <Bar dataKey="nossoSGD" name="Nosso SGD" fill="#10b981" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="artigoSGD" name="Artigo SGD" fill="#059669" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="nossoAdam" name="Nosso Adam" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="artigoAdam" name="Artigo Adam" fill="#d97706" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                  Perfil Multidimensional de Desempenho
                </h3>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="#334155" />
                      <PolarAngleAxis dataKey="metric" stroke="#94a3b8" fontSize={12} />
                      <PolarRadiusAxis domain={[0, 100]} stroke="#475569" />
                      <Radar name="SGD (Otimizado)" dataKey="SGD" stroke="#10b981" fill="#10b981" fillOpacity={0.4} />
                      <Radar name="Adam (Colapso)" dataKey="Adam" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.4} />
                      <Legend />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8 }} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: SQUEEZENET */}
        {activeTab === 'squeezenet' && (
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                <Zap className="w-5 h-5 text-emerald-400" />
                SqueezeNet 1.1: Eficiência em Borda e Dispositivos Móveis
              </h2>
              <p className="text-sm text-slate-400 max-w-3xl mb-6">
                A SqueezeNet foi projetada para oferecer acurácia comparável a redes densas com uma redução drástica no número de parâmetros usando módulos <em>Fire</em>.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5">
                  <h3 className="text-white font-semibold text-sm mb-3">Comparação de Peso de Modelo (.pth)</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300 font-medium">VGG-16 (Adam/SGD)</span>
                        <span className="text-red-400 font-mono font-bold">~537 MB</span>
                      </div>
                      <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                        <div className="bg-red-500 h-2.5 rounded-full" style={{ width: '100%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300 font-medium">ResNet-50</span>
                        <span className="text-amber-400 font-mono font-bold">~94 MB</span>
                      </div>
                      <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                        <div className="bg-amber-500 h-2.5 rounded-full" style={{ width: '17.5%' }}></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-300 font-medium">SqueezeNet 1.1</span>
                        <span className="text-emerald-400 font-mono font-bold">~2.9 MB</span>
                      </div>
                      <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
                        <div className="bg-emerald-500 h-2.5 rounded-full" style={{ width: '1.2%' }}></div>
                      </div>
                    </div>
                  </div>
                  <p className="text-xs text-slate-400 mt-4">
                    A SqueezeNet é mais de <strong>180 vezes menor</strong> que a VGG-16, permitindo execução em hardware embarcado de laboratórios de patologia.
                  </p>
                </div>

                <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-5">
                  <h3 className="text-white font-semibold text-sm mb-3">Treinamentos Realizados</h3>
                  <div className="space-y-2.5 text-xs">
                    <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 flex items-center justify-between">
                      <span className="text-slate-200">squeezenet_original_adam.pth</span>
                      <span className="text-emerald-400 font-mono font-medium">Concluído (2.9 MB)</span>
                    </div>
                    <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 flex items-center justify-between">
                      <span className="text-slate-200">main_squeezenet.py (SGD / Adam)</span>
                      <span className="text-indigo-400 font-mono font-medium">Script Modular</span>
                    </div>
                    <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 flex items-center justify-between">
                      <span className="text-slate-200">treinamento_squeezenet_colab.ipynb</span>
                      <span className="text-amber-400 font-mono font-medium">Caderno Colab</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <p>Projeto de Processamento Digital de Imagens (PDI) · Replicação de Artigo & Aplicação Web</p>
      </footer>
    </div>
  )
}
