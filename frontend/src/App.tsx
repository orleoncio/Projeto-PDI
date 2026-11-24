import { useState } from 'react'
import {
  Activity,
  Layers,
  FileText,
  BarChart3,
  AlertTriangle,
  TrendingUp,
  Brain,
  Zap,
  Info,
  Sparkles,
  Scale,
  Award,
  Globe
} from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts'

// Dados dos 6 cenários da VGG-16 (Artigo vs Nosso Treino)
const vgg16Consolidado = [
  {
    entrada: 'Original',
    otimizador: 'SGD',
    artigoAcc: 98.16,
    nossoAcc: 97.98,
    deltaAcc: -0.18,
    artigoF1: 98.05,
    nossoF1: 97.64,
    deltaF1: -0.41,
    artigoPrec: 98.14,
    nossoPrec: 97.56,
    artigoRec: 97.98,
    nossoRec: 97.73,
    status: 'Sucesso Absoluto',
    destaque: 'Convergência estável via Data Augmentation D4',
    badge: 'success'
  },
  {
    entrada: 'Original',
    otimizador: 'ADAM',
    artigoAcc: 19.96,
    nossoAcc: 21.73,
    deltaAcc: 1.77,
    artigoF1: 5.55,
    nossoF1: 5.95,
    deltaF1: 0.40,
    artigoPrec: 3.33,
    nossoPrec: 3.62,
    artigoRec: 16.67,
    nossoRec: 16.67,
    status: 'Colapso Reproduzido',
    destaque: 'Loss travada em 1.728; predição de classe única (1/6 = 16.67%)',
    badge: 'danger'
  },
  {
    entrada: 'Contraste',
    otimizador: 'SGD',
    artigoAcc: 97.76,
    nossoAcc: 97.57,
    deltaAcc: -0.19,
    artigoF1: 97.58,
    nossoF1: 97.25,
    deltaF1: -0.33,
    artigoPrec: 97.60,
    nossoPrec: 97.27,
    artigoRec: 97.56,
    nossoRec: 97.24,
    status: 'Sucesso Absoluto',
    destaque: 'Diferença inferior a 0.2 p.p. na acurácia',
    badge: 'success'
  },
  {
    entrada: 'Contraste',
    otimizador: 'ADAM',
    artigoAcc: 86.32,
    nossoAcc: 87.06,
    deltaAcc: 0.74,
    artigoF1: 86.35,
    nossoF1: 86.76,
    deltaF1: 0.41,
    artigoPrec: 86.69,
    nossoPrec: 88.43,
    artigoRec: 86.69,
    nossoRec: 85.60,
    status: 'Resgate Comprovado',
    destaque: 'Fim do colapso: expansão de contraste resgata convergência',
    badge: 'highlight'
  },
  {
    entrada: 'Equalização',
    otimizador: 'SGD',
    artigoAcc: 96.80,
    nossoAcc: 97.39,
    deltaAcc: 0.59,
    artigoF1: 96.73,
    nossoF1: 97.12,
    deltaF1: 0.39,
    artigoPrec: 96.76,
    nossoPrec: 97.28,
    artigoRec: 96.71,
    nossoRec: 96.97,
    status: 'Sucesso Absoluto',
    destaque: 'SGD mantém robustez superior mesmo com equalização',
    badge: 'success'
  },
  {
    entrada: 'Equalização',
    otimizador: 'ADAM',
    artigoAcc: 20.81,
    nossoAcc: 21.73,
    deltaAcc: 0.92,
    artigoF1: 5.74,
    nossoF1: 5.95,
    deltaF1: 0.21,
    artigoPrec: 3.47,
    nossoPrec: 3.62,
    artigoRec: 16.67,
    nossoRec: 16.67,
    status: 'Colapso Reproduzido',
    destaque: 'Equalização achata gradientes e prende Adam em platô estocástico',
    badge: 'danger'
  },
]

// Dados dos 6 cenários da SqueezeNet 1.1 (Projeto vs Artigo SqueezeNet 1.0)
const squeezenetResultados = [
  {
    entrada: 'Original',
    otimizador: 'Adam',
    artigoAcc: 93.90,
    projetoAcc: 92.35,
    deltaAcc: -1.55,
    artigoF1: 93.43,
    projetoF1: 92.15,
    deltaF1: -1.28,
    artigoPrec: 93.47,
    projetoPrec: 91.78,
    artigoRec: 93.45,
    projetoRec: 92.84,
    observacao: 'Melhor resultado geral da SqueezeNet; recall superior a precision.'
  },
  {
    entrada: 'Original',
    otimizador: 'SGD',
    artigoAcc: 96.40,
    projetoAcc: 89.93,
    deltaAcc: -6.47,
    artigoF1: 96.18,
    projetoF1: 90.25,
    deltaF1: -5.93,
    artigoPrec: 96.82,
    projetoPrec: 89.78,
    artigoRec: 95.66,
    projetoRec: 90.96,
    observacao: 'Convergência satisfatória (~90%), porém inferior à referência.'
  },
  {
    entrada: 'Contraste',
    otimizador: 'Adam',
    artigoAcc: 92.68,
    projetoAcc: 91.32,
    deltaAcc: -1.36,
    artigoF1: 92.09,
    projetoF1: 91.05,
    deltaF1: -1.04,
    artigoPrec: 92.75,
    projetoPrec: 92.08,
    artigoRec: 91.65,
    projetoRec: 90.73,
    observacao: 'Precision quase idêntica à referência (-0.67 p.p.).'
  },
  {
    entrada: 'Contraste',
    otimizador: 'SGD',
    artigoAcc: 96.47,
    projetoAcc: 89.60,
    deltaAcc: -6.87,
    artigoF1: 96.35,
    projetoF1: 90.14,
    deltaF1: -6.21,
    artigoPrec: 96.39,
    projetoPrec: 91.21,
    artigoRec: 96.33,
    projetoRec: 89.41,
    observacao: 'Próximo do cenário original com SGD.'
  },
  {
    entrada: 'Equalização',
    otimizador: 'Adam',
    artigoAcc: 86.62,
    projetoAcc: 87.43,
    deltaAcc: 0.81,
    artigoF1: 86.53,
    projetoF1: 87.78,
    deltaF1: 1.25,
    artigoPrec: 87.79,
    projetoPrec: 88.48,
    artigoRec: 86.48,
    projetoRec: 87.24,
    observacao: 'Superou ligeiramente os 4 valores de referência do artigo!'
  },
  {
    entrada: 'Equalização',
    otimizador: 'SGD',
    artigoAcc: 96.29,
    projetoAcc: 80.00,
    deltaAcc: -16.29,
    artigoF1: 96.05,
    projetoF1: 68.79,
    deltaF1: -27.26,
    artigoPrec: 96.37,
    projetoPrec: 67.38,
    artigoRec: 95.78,
    projetoRec: 71.19,
    observacao: 'Pior cenário medido: queda severa em precision (67.38%) e F1 (68.79%).'
  },
]

// Gráfico comparativo SqueezeNet 1.1
const squeezenetChartData = [
  { cenario: 'Orig. Adam', Acuracia: 92.35, F1: 92.15 },
  { cenario: 'Orig. SGD', Acuracia: 89.93, F1: 90.25 },
  { cenario: 'Contr. Adam', Acuracia: 91.32, F1: 91.05 },
  { cenario: 'Contr. SGD', Acuracia: 89.60, F1: 90.14 },
  { cenario: 'Eq. Adam', Acuracia: 87.43, F1: 87.78 },
  { cenario: 'Eq. SGD', Acuracia: 80.00, F1: 68.79 },
]

// Tabela 3 Geral Multi-Modelos do Artigo
const tabelaGeralArtigo = [
  {
    modelo: 'AlexNet',
    adamContr: '89.93% / 89.79%',
    adamEq: '22.32% / 6.08% (Colapso)',
    adamOrig: '21.69% / 5.95% (Colapso)',
    sgdContr: '96.65% / 96.41%',
    sgdEq: '95.22% / 95.07%',
    sgdOrig: '95.99% / 95.92%',
  },
  {
    modelo: 'Inception V3',
    adamContr: '96.65% / 96.38%',
    adamEq: '95.04% / 94.64%',
    adamOrig: '97.54% / 97.45%',
    sgdContr: '96.54% / 96.27%',
    sgdEq: '95.29% / 94.95%',
    sgdOrig: '97.83% / 97.45%',
  },
  {
    modelo: 'ResNet-50',
    adamContr: '96.47% / 96.18%',
    adamEq: '94.71% / 94.69%',
    adamOrig: '97.94% / 97.57%',
    sgdContr: '94.63% / 94.07%',
    sgdEq: '92.57% / 92.39%',
    sgdOrig: '96.88% / 96.59%',
  },
  {
    modelo: 'SqueezeNet 1.0 (Artigo)',
    adamContr: '92.68% / 92.09%',
    adamEq: '86.62% / 86.53%',
    adamOrig: '93.90% / 93.43%',
    sgdContr: '96.47% / 96.35%',
    sgdEq: '96.29% / 96.05%',
    sgdOrig: '96.40% / 96.18%',
  },
  {
    modelo: 'VGG-16 (Artigo)',
    adamContr: '86.32% / 86.35% (Resgate)',
    adamEq: '20.81% / 5.74% (Colapso)',
    adamOrig: '19.96% / 5.55% (Colapso)',
    sgdContr: '97.76% / 97.58%',
    sgdEq: '96.80% / 96.73%',
    sgdOrig: '98.16% / 98.05%',
  },
  {
    modelo: 'VGG-16 (Nosso Treino)',
    adamContr: '87.06% / 86.76% (Resgate)',
    adamEq: '21.73% / 5.95% (Colapso)',
    adamOrig: '21.73% / 5.95% (Colapso)',
    sgdContr: '97.57% / 97.25%',
    sgdEq: '97.39% / 97.12%',
    sgdOrig: '97.98% / 97.64%',
    destaque: true
  },
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
  const [activeTab, setActiveTab] = useState<'visao-geral' | 'descobertas' | 'vgg16' | 'squeezenet' | 'tabela-geral'>('descobertas')
  const [vggFilter, setVggFilter] = useState<'todos' | 'sgd' | 'adam'>('todos')

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
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
                Apresentação Interativa de Resultados & Descobertas
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
      <section className="bg-gradient-to-b from-slate-900 via-slate-900/80 to-slate-950 border-b border-slate-800/80 py-8 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-medium mb-3">
            <FileText className="w-3.5 h-3.5" />
            RITA (Vol. 32, 2025): Preprocessing and Optimizer Impact on Image-Based Autoimmune Disease Diagnosis
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-white max-w-4xl">
            Descobertas Científicas e Resultados Experimentais
          </h1>
          <p className="mt-2 text-slate-400 max-w-3xl text-sm sm:text-base leading-relaxed">
            Replicação empírica e confronto dos 6 cenários da VGG-16, dos 6 cenários da SqueezeNet e investigação profunda sobre o motivo do colapso do Adam vs. Transfer Learning.
          </p>

          {/* Navigation Tabs */}
          <div className="flex flex-wrap gap-2 mt-6 border-b border-slate-800 pb-px">
            <button
              onClick={() => setActiveTab('descobertas')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-semibold transition-all ${
                activeTab === 'descobertas'
                  ? 'bg-slate-900 text-amber-400 border-b-2 border-amber-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Sparkles className="w-4 h-4 text-amber-400" />
              🔬 Descobertas Científicas & Artigo
            </button>
            <button
              onClick={() => setActiveTab('vgg16')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-semibold transition-all ${
                activeTab === 'vgg16'
                  ? 'bg-slate-900 text-emerald-400 border-b-2 border-emerald-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              📊 VGG-16 (6 Cenários)
            </button>
            <button
              onClick={() => setActiveTab('squeezenet')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-semibold transition-all ${
                activeTab === 'squeezenet'
                  ? 'bg-slate-900 text-sky-400 border-b-2 border-sky-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Zap className="w-4 h-4 text-sky-400" />
              ⚡ SqueezeNet (6 Cenários)
            </button>
            <button
              onClick={() => setActiveTab('tabela-geral')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-semibold transition-all ${
                activeTab === 'tabela-geral'
                  ? 'bg-slate-900 text-purple-400 border-b-2 border-purple-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Globe className="w-4 h-4 text-purple-400" />
              🌐 Tabela 3 Geral (Todas as Redes)
            </button>
            <button
              onClick={() => setActiveTab('visao-geral')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg text-sm font-semibold transition-all ${
                activeTab === 'visao-geral'
                  ? 'bg-slate-900 text-emerald-400 border-b-2 border-emerald-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
              }`}
            >
              <Info className="w-4 h-4" />
              Classes HEp-2 & Contexto
            </button>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ============================================================ */}
        {/* TAB 1: DESCOBERTAS CIENTÍFICAS */}
        {/* ============================================================ */}
        {activeTab === 'descobertas' && (
          <div className="space-y-8">
            {/* Banner de Destaque da Descoberta */}
            <div className="bg-gradient-to-r from-amber-500/10 via-slate-900 to-indigo-500/10 border border-amber-500/30 rounded-2xl p-6 sm:p-8">
              <div className="flex items-center gap-3 mb-3">
                <span className="p-2 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/30">
                  <Sparkles className="w-6 h-6" />
                </span>
                <div>
                  <h2 className="text-xl font-bold text-white">Nossa Principal Descoberta Científica</h2>
                  <p className="text-xs text-amber-300">Treinamento do Zero (weights=None) vs. Transfer Learning (ImageNet)</p>
                </div>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed max-w-4xl">
                O artigo de referência reporta um colapso severo do otimizador <strong>Adam</strong> na VGG-16 (~19.96% de acurácia). 
                Ao replicarmos a metodologia, descobrimos exatamente o <strong>porquê</strong> desse fenômeno acontecer e provamos como o <em>Transfer Learning</em> elimina a vulnerabilidade.
              </p>
            </div>

            {/* Grid dos dois cenários */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Cenário A: Treino do Zero */}
              <div className="bg-slate-900/80 border border-amber-500/40 rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 px-3 py-1 bg-amber-500/20 text-amber-400 text-xs font-semibold rounded-bl-lg border-l border-b border-amber-500/30">
                  Metodologia Fiel ao Artigo
                </div>
                <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-amber-400" />
                  Cenário A: Treino do Zero (<code className="text-xs bg-slate-800 px-1.5 py-0.5 rounded text-amber-300">weights=None</code>)
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  138 milhões de parâmetros inicializados aleatoriamente, sem camadas de <em>Batch Normalization</em>.
                </p>
                <div className="space-y-3 text-xs">
                  <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
                    <span className="font-semibold text-emerald-400 block mb-1">SGD com Momentum:</span>
                    Convergiu de forma estável atingindo <strong>97.98% de acurácia</strong> (artigo: 98.16%). A taxa de aprendizado única global com inércia física evitou instabilidades.
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950/80 border border-amber-500/20">
                    <span className="font-semibold text-amber-400 block mb-1">Colapso Matemático do Adam:</span>
                    A perda estagnou em <code className="text-amber-300">Loss ≈ 1.728 ≈ ln(6)</code>. O Adam previu 100% das imagens na classe majoritária (*Centromere*), fixando o Recall em exatamente <code className="text-amber-300">1/6 = 16.67%</code> e acurácia de <strong>21.73%</strong> (artigo: 19.96%).
                  </div>
                </div>
              </div>

              {/* Cenário B: Transfer Learning */}
              <div className="bg-slate-900/80 border border-emerald-500/40 rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 px-3 py-1 bg-emerald-500/20 text-emerald-400 text-xs font-semibold rounded-bl-lg border-l border-b border-emerald-500/30">
                  Experimento Investigativo
                </div>
                <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                  <Award className="w-5 h-5 text-emerald-400" />
                  Cenário B: Transfer Learning (<code className="text-xs bg-slate-800 px-1.5 py-0.5 rounded text-emerald-300">weights="DEFAULT"</code>)
                </h3>
                <p className="text-xs text-slate-400 mb-4">
                  Pesos pré-treinados no ImageNet com extratores consolidados de texturas e bordas.
                </p>
                <div className="space-y-3 text-xs">
                  <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
                    <span className="font-semibold text-emerald-400 block mb-1">O Adam NÃO colapsa:</span>
                    Com pré-condicionamento dos pesos, o Adam saltou de 52.89% na Época 1 para mais de <strong>86.5% na Época 9</strong>, provando que o colapso não é um problema do dataset, mas da inicialização aleatória.
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800">
                    <span className="font-semibold text-emerald-400 block mb-1">Conclusão Metodológica:</span>
                    Para replicar fielmente o artigo e medir o impacto do otimizador, era obrigatório treinar do zero. Se usássemos transfer learning, mascararíamos o fenômeno medido pelos autores!
                  </div>
                </div>
              </div>
            </div>

            {/* Tabela do Experimento de Transfer Learning */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-base font-bold text-white mb-3 flex items-center gap-2">
                <Scale className="w-5 h-5 text-indigo-400" />
                Quadro Comparativo: Do Zero vs. Transfer Learning (Imagem Original)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-4">Modelo / Otimizador</th>
                      <th className="py-2.5 px-4">Inicialização</th>
                      <th className="py-2.5 px-4">Épocas</th>
                      <th className="py-2.5 px-4">Accuracy Teste</th>
                      <th className="py-2.5 px-4">F1-Score</th>
                      <th className="py-2.5 px-4">Diagnóstico Metodológico</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    <tr className="hover:bg-slate-800/20">
                      <td className="py-2.5 px-4 font-sans font-bold text-slate-200">VGG-16 + SGD</td>
                      <td className="py-2.5 px-4 text-amber-300">Do zero (None)</td>
                      <td className="py-2.5 px-4">30</td>
                      <td className="py-2.5 px-4 text-emerald-400 font-bold">89.67% ~ 97.98%</td>
                      <td className="py-2.5 px-4 text-emerald-400">89.79% ~ 97.64%</td>
                      <td className="py-2.5 px-4 font-sans text-slate-300">Convergência padrão e consistente</td>
                    </tr>
                    <tr className="hover:bg-slate-800/20">
                      <td className="py-2.5 px-4 font-sans font-bold text-slate-200">VGG-16 + SGD</td>
                      <td className="py-2.5 px-4 text-emerald-300">Transfer Learning (ImageNet)</td>
                      <td className="py-2.5 px-4">15</td>
                      <td className="py-2.5 px-4 text-emerald-400 font-bold">90.51%</td>
                      <td className="py-2.5 px-4 text-emerald-400">90.57%</td>
                      <td className="py-2.5 px-4 font-sans text-slate-300">Convergência ultra-rápida com GPU</td>
                    </tr>
                    <tr className="hover:bg-slate-800/20 bg-amber-500/5">
                      <td className="py-2.5 px-4 font-sans font-bold text-amber-300">VGG-16 + ADAM</td>
                      <td className="py-2.5 px-4 text-amber-300">Do zero (None)</td>
                      <td className="py-2.5 px-4">30</td>
                      <td className="py-2.5 px-4 text-amber-400 font-bold">21.73%</td>
                      <td className="py-2.5 px-4 text-amber-400">5.95%</td>
                      <td className="py-2.5 px-4 font-sans text-amber-300">Colapso total por desvanecimento de gradientes</td>
                    </tr>
                    <tr className="hover:bg-slate-800/20 bg-emerald-500/5">
                      <td className="py-2.5 px-4 font-sans font-bold text-emerald-300">VGG-16 + ADAM</td>
                      <td className="py-2.5 px-4 text-emerald-300">Transfer Learning (ImageNet)</td>
                      <td className="py-2.5 px-4">15</td>
                      <td className="py-2.5 px-4 text-emerald-400 font-bold">~86.50%</td>
                      <td className="py-2.5 px-4 text-emerald-400">~86.20%</td>
                      <td className="py-2.5 px-4 font-sans text-emerald-300">Resgate do modelo sem necessidade de PDI</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Outras 2 grandes descobertas */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
                <h4 className="font-bold text-white mb-2 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-indigo-400" />
                  O Fenômeno do Resgate pelo Contraste
                </h4>
                <p className="text-xs text-slate-300 leading-relaxed mb-3">
                  Quando mantido o treino do zero, a <strong>Expansão de Contraste</strong> atuou como um regularizador do espaço de características, resgatando a VGG-16 com Adam de <strong>21.73% para 87.06%</strong> de acurácia (artigo: 86.32%).
                </p>
                <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs font-mono text-slate-400">
                  Adam Original: 21.73% Acc → Adam Contraste: 87.06% Acc (+65.33 p.p.)
                </div>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
                <h4 className="font-bold text-white mb-2 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-purple-400" />
                  Análise Fatorial do Artigo (Tabelas 6 e 7)
                </h4>
                <p className="text-xs text-slate-300 leading-relaxed mb-3">
                  O artigo realizou análise de variância fatorial demonstrando que a VGG-16 é a arquitetura mais sensível ao otimizador entre todas: <strong>47.11% da variância</strong> é explicada pelo otimizador, tornando-a o estudo de caso perfeito.
                </p>
                <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 text-xs font-mono text-slate-400">
                  Dependência do Otimizador: VGG-16 = 47.11% (Maior entre todas as CNNs)
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 2: VGG-16 (6 CENÁRIOS CONSOLIDADOS) */}
        {/* ============================================================ */}
        {activeTab === 'vgg16' && (
          <div className="space-y-8">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
                <div>
                  <h2 className="text-lg font-bold text-white flex items-center gap-2">
                    <Activity className="w-5 h-5 text-emerald-400" />
                    Consolidação dos 6 Cenários da VGG-16 (Artigo vs. Nosso Treinamento)
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Confrontação direta com a Tabela 3 do artigo: erro médio absoluto de apenas <strong>0.73% em Accuracy</strong> e <strong>0.36% em F1</strong>.
                  </p>
                </div>
                <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                  <button
                    onClick={() => setVggFilter('todos')}
                    className={`px-3 py-1 rounded-md transition ${vggFilter === 'todos' ? 'bg-slate-800 text-white font-medium' : 'text-slate-400'}`}
                  >
                    Todos (6)
                  </button>
                  <button
                    onClick={() => setVggFilter('sgd')}
                    className={`px-3 py-1 rounded-md transition ${vggFilter === 'sgd' ? 'bg-emerald-500/20 text-emerald-300 font-medium' : 'text-slate-400'}`}
                  >
                    Apenas SGD
                  </button>
                  <button
                    onClick={() => setVggFilter('adam')}
                    className={`px-3 py-1 rounded-md transition ${vggFilter === 'adam' ? 'bg-amber-500/20 text-amber-300 font-medium' : 'text-slate-400'}`}
                  >
                    Apenas Adam
                  </button>
                </div>
              </div>

              {/* Tabela dos 6 Cenários */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-3 px-3">Entrada</th>
                      <th className="py-3 px-3">Otimizador</th>
                      <th className="py-3 px-3">Artigo Acc</th>
                      <th className="py-3 px-3">Nosso Acc</th>
                      <th className="py-3 px-3">Δ Acc</th>
                      <th className="py-3 px-3">Artigo F1</th>
                      <th className="py-3 px-3">Nosso F1</th>
                      <th className="py-3 px-3">Δ F1</th>
                      <th className="py-3 px-3">Veredito da Replicação</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {vgg16Consolidado
                      .filter(item => vggFilter === 'todos' || item.otimizador.toLowerCase() === vggFilter)
                      .map((item, idx) => (
                        <tr key={idx} className="hover:bg-slate-800/30 transition">
                          <td className="py-3 px-3 font-sans font-semibold text-slate-200">{item.entrada}</td>
                          <td className="py-3 px-3">
                            <span className={`px-2 py-0.5 rounded font-bold ${item.otimizador === 'SGD' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
                              {item.otimizador}
                            </span>
                          </td>
                          <td className="py-3 px-3 text-slate-300">{item.artigoAcc.toFixed(2)}%</td>
                          <td className="py-3 px-3 text-white font-bold">{item.nossoAcc.toFixed(2)}%</td>
                          <td className={`py-3 px-3 font-bold ${item.deltaAcc >= 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                            {item.deltaAcc >= 0 ? `+${item.deltaAcc.toFixed(2)}%` : `${item.deltaAcc.toFixed(2)}%`}
                          </td>
                          <td className="py-3 px-3 text-slate-300">{item.artigoF1.toFixed(2)}%</td>
                          <td className="py-3 px-3 text-white font-bold">{item.nossoF1.toFixed(2)}%</td>
                          <td className={`py-3 px-3 font-bold ${item.deltaF1 >= 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                            {item.deltaF1 >= 0 ? `+${item.deltaF1.toFixed(2)}%` : `${item.deltaF1.toFixed(2)}%`}
                          </td>
                          <td className="py-3 px-3 font-sans">
                            <div className="flex flex-col">
                              <span className={`inline-block font-semibold ${
                                item.badge === 'success' ? 'text-emerald-400' : item.badge === 'danger' ? 'text-amber-400' : 'text-sky-400'
                              }`}>
                                {item.status}
                              </span>
                              <span className="text-[10px] text-slate-400">{item.destaque}</span>
                            </div>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Gráficos Comparativos VGG-16 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-emerald-400" />
                  Acurácia nos 6 Cenários VGG-16 (Artigo vs Nosso Treino)
                </h3>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={vgg16Consolidado} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="entrada" stroke="#94a3b8" fontSize={11} tickFormatter={(val, i) => `${val} (${vgg16Consolidado[i]?.otimizador})`} />
                      <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 100]} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8 }} />
                      <Legend />
                      <Bar dataKey="nossoAcc" name="Nosso Treino (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="artigoAcc" name="Artigo Oficial (%)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-amber-400" />
                  F1-Score nos 6 Cenários VGG-16
                </h3>
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={vgg16Consolidado} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="entrada" stroke="#94a3b8" fontSize={11} tickFormatter={(val, i) => `${val} (${vgg16Consolidado[i]?.otimizador})`} />
                      <YAxis stroke="#94a3b8" fontSize={11} domain={[0, 100]} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8 }} />
                      <Legend />
                      <Bar dataKey="nossoF1" name="Nosso F1 (%)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="artigoF1" name="Artigo F1 (%)" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 3: SQUEEZENET 1.1 (6 CENÁRIOS COMPLETOS) */}
        {/* ============================================================ */}
        {activeTab === 'squeezenet' && (
          <div className="space-y-8">
            {/* Nota Metodológica SqueezeNet 1.0 vs 1.1 */}
            <div className="bg-sky-500/10 border border-sky-500/30 rounded-2xl p-6">
              <div className="flex items-center gap-3 mb-2">
                <span className="p-2 rounded-xl bg-sky-500/20 text-sky-400 border border-sky-500/30">
                  <Zap className="w-5 h-5" />
                </span>
                <div>
                  <h3 className="text-base font-bold text-white">Consolidação Experimental da SqueezeNet</h3>
                  <p className="text-xs text-sky-300">SqueezeNet 1.1 (TorchVision, nosso projeto) vs. SqueezeNet 1.0 (Artigo)</p>
                </div>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed max-w-4xl">
                O artigo utilizou a <strong>SqueezeNet 1.0</strong>, enquanto instanciamos a <strong>SqueezeNet 1.1</strong> do TorchVision (mais leve e com menos parâmetros). As métricas publicadas servem como referência contextual. O Adam obteve o melhor desempenho geral na SqueezeNet, atingindo <strong>92.35% de acurácia</strong> na imagem original.
              </p>
            </div>

            {/* Tabela Completa dos 6 Cenários da SqueezeNet */}
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-sky-400" />
                Tabela Comparativa Geral: Artigo vs. Checkpoints do Colab
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-3 px-3">Entrada</th>
                      <th className="py-3 px-3">Otimizador</th>
                      <th className="py-3 px-3">Métrica</th>
                      <th className="py-3 px-3">Artigo (1.0)</th>
                      <th className="py-3 px-3">Projeto (1.1)</th>
                      <th className="py-3 px-3">Δ (p.p.)</th>
                      <th className="py-3 px-3">Observação Científica</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {squeezenetResultados.map((item, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/30 transition">
                        <td className="py-3 px-3 font-sans font-semibold text-slate-200">{item.entrada}</td>
                        <td className="py-3 px-3">
                          <span className={`px-2 py-0.5 rounded font-bold ${item.otimizador === 'Adam' ? 'bg-sky-500/20 text-sky-300' : 'bg-emerald-500/20 text-emerald-400'}`}>
                            {item.otimizador}
                          </span>
                        </td>
                        <td className="py-3 px-3 font-sans text-slate-300">
                          <div>Accuracy</div>
                          <div>Precision</div>
                          <div>Recall</div>
                          <div>F1-score</div>
                        </td>
                        <td className="py-3 px-3 text-slate-400">
                          <div>{item.artigoAcc.toFixed(2)}%</div>
                          <div>{item.artigoPrec.toFixed(2)}%</div>
                          <div>{item.artigoRec.toFixed(2)}%</div>
                          <div>{item.artigoF1.toFixed(2)}%</div>
                        </td>
                        <td className="py-3 px-3 text-white font-bold">
                          <div className="text-sky-400">{item.projetoAcc.toFixed(2)}%</div>
                          <div>{item.projetoPrec.toFixed(2)}%</div>
                          <div>{item.projetoRec.toFixed(2)}%</div>
                          <div className="text-emerald-400">{item.projetoF1.toFixed(2)}%</div>
                        </td>
                        <td className="py-3 px-3">
                          <div className={item.deltaAcc >= 0 ? 'text-emerald-400' : 'text-slate-400'}>
                            {item.deltaAcc >= 0 ? `+${item.deltaAcc.toFixed(2)}` : item.deltaAcc.toFixed(2)}
                          </div>
                          <div>{item.deltaAcc >= 0 ? `+${(item.projetoPrec - item.artigoPrec).toFixed(2)}` : (item.projetoPrec - item.artigoPrec).toFixed(2)}</div>
                          <div>{item.deltaAcc >= 0 ? `+${(item.projetoRec - item.artigoRec).toFixed(2)}` : (item.projetoRec - item.artigoRec).toFixed(2)}</div>
                          <div className={item.deltaF1 >= 0 ? 'text-emerald-400' : 'text-slate-400'}>
                            {item.deltaF1 >= 0 ? `+${item.deltaF1.toFixed(2)}` : item.deltaF1.toFixed(2)}
                          </div>
                        </td>
                        <td className="py-3 px-3 font-sans text-slate-300 text-[11px] max-w-xs">
                          {item.observacao}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Tabela Resumida & Gráfico SqueezeNet */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
                <h4 className="font-bold text-white text-sm mb-3">Tabela Resumida da SqueezeNet 1.1</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="py-2 px-3 font-sans">Pré-processamento</th>
                        <th className="py-2 px-3 font-sans">Otimizador</th>
                        <th className="py-2 px-3 font-sans">Accuracy</th>
                        <th className="py-2 px-3 font-sans">F1 Macro</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {squeezenetResultados.map((r, i) => (
                        <tr key={i} className="hover:bg-slate-800/20">
                          <td className="py-2 px-3 font-sans text-slate-200">{r.entrada}</td>
                          <td className="py-2 px-3 text-sky-400">{r.otimizador}</td>
                          <td className="py-2 px-3 font-bold text-white">{r.projetoAcc.toFixed(2)}%</td>
                          <td className="py-2 px-3 font-bold text-emerald-400">{r.projetoF1.toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
                <h4 className="font-bold text-white text-sm mb-4">Desempenho por Cenário na SqueezeNet 1.1 (%)</h4>
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={squeezenetChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="cenario" stroke="#94a3b8" fontSize={10} />
                      <YAxis stroke="#94a3b8" fontSize={11} domain={[60, 100]} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: 8 }} />
                      <Legend />
                      <Bar dataKey="Acuracia" name="Accuracy (%)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="F1" name="F1 Macro (%)" fill="#34d399" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Destaques da SqueezeNet */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                <span className="font-semibold text-sky-400 block mb-1">Adam Lidera:</span>
                Ao contrário da VGG-16, na SqueezeNet o Adam teve o melhor resultado geral (92.35% Acc / 92.15% F1 na imagem original).
              </div>
              <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                <span className="font-semibold text-amber-400 block mb-1">Anomalia em Equalização + SGD:</span>
                Queda acentuada: accuracy de 80.00% e F1 de 68.79%, indicando que a equalização forçada prejudicou fortemente os filtros convolucionais compactos.
              </div>
              <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                <span className="font-semibold text-emerald-400 block mb-1">Pegada de Memória Mínima:</span>
                O checkpoint da SqueezeNet tem apenas <strong>~2.9 MB</strong> contra 537 MB da VGG-16 (~180x menor), sendo ideal para sistemas clínicos de borda.
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 4: TABELA 3 GERAL (TODAS AS REDES DO ARTIGO) */}
        {/* ============================================================ */}
        {activeTab === 'tabela-geral' && (
          <div className="space-y-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                <Globe className="w-5 h-5 text-purple-400" />
                Tabela 3 na Íntegra (Artigo Oficial RITA 2025 vs. Nosso Projeto)
              </h2>
              <p className="text-xs text-slate-400 mb-6">
                Todas as arquiteturas avaliadas pelos autores (AlexNet, Inception V3, ResNet-50, SqueezeNet e VGG-16) nos três cenários de pré-processamento.
              </p>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 text-slate-400 uppercase tracking-wider border-b border-slate-800">
                    <tr>
                      <th className="py-3 px-3">Arquitetura</th>
                      <th className="py-3 px-3">Otimizador</th>
                      <th className="py-3 px-3">Expansão de Contraste (Acc / F1)</th>
                      <th className="py-3 px-3">Equalização Histograma (Acc / F1)</th>
                      <th className="py-3 px-3">Imagem Original (Acc / F1)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {tabelaGeralArtigo.map((item, idx) => (
                      <tr key={idx} className={`hover:bg-slate-800/30 transition ${item.destaque ? 'bg-emerald-500/10 border-l-4 border-emerald-400' : ''}`}>
                        <td className="py-3 px-3 font-sans font-bold text-slate-200">
                          {item.modelo}
                          {item.destaque && (
                            <span className="block text-[10px] text-emerald-400 font-normal">★ Nossa Replicação Fiel</span>
                          )}
                        </td>
                        <td className="py-3 px-3">
                          <div className="text-amber-400 font-bold mb-1">ADAM</div>
                          <div className="text-emerald-400 font-bold">SGD</div>
                        </td>
                        <td className="py-3 px-3">
                          <div className="text-slate-300 mb-1">{item.adamContr}</div>
                          <div className="text-slate-300">{item.sgdContr}</div>
                        </td>
                        <td className="py-3 px-3">
                          <div className="text-slate-300 mb-1">{item.adamEq}</div>
                          <div className="text-slate-300">{item.sgdEq}</div>
                        </td>
                        <td className="py-3 px-3">
                          <div className="text-slate-300 mb-1">{item.adamOrig}</div>
                          <div className="text-slate-300">{item.sgdOrig}</div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* TAB 5: CLASSES HEP-2 & CONTEXTO */}
        {/* ============================================================ */}
        {activeTab === 'visao-geral' && (
          <div className="space-y-8">
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
              <h2 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
                <Layers className="w-5 h-5 text-emerald-400" />
                Classes Celulares do Dataset HEp-2 (ICPR 2014)
              </h2>
              <p className="text-sm text-slate-400 mb-6">
                Conjunto de dados composto por 13.596 imagens de imunofluorescência indireta para diagnóstico de doenças autoimunes:
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
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <p>Projeto PDI · Replicação do Artigo RITA (2025) · Aplicação Web de Apresentação</p>
      </footer>
    </div>
  )
}
