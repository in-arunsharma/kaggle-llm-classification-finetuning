# Predicció de Preferències en Respostes de Models de Llenguatge

## Autors
**Iker Bolancel** | **Arun Sharma**  
Enginyeria Informàtica  
Universitat Autònoma de Barcelona  
Assignatura: Aprenentatge Computacional  
Desembre 2025

---

## Descripció del Projecte

Aquest treball aborda el problema de la predicció automàtica de preferències humanes entre respostes generades per diferents models de llenguatge. L'objectiu principal consisteix en desenvolupar un sistema capaç de determinar quina de dues respostes generades per diferents models de llenguatge és preferida pels usuaris humans, o si ambdues respostes són equivalents.

El conjunt de dades emprat conté prompts originals, respostes generades per diversos models de llenguatge (GPT, Claude, Gemini, Llama, Mistral, entre d'altres), i etiquetes que indiquen la preferència humana entre elles. Aquesta tasca representa un desafiament complex en l'àmbit del Processament de Llenguatge Natural, ja que implica replicar judicis subjectius humans sobre qualitat textual.

## Metodologia

### Anàlisi Exploratòria de Dades

L'anàlisi inicial del dataset revela diverses característiques rellevants que han guiat el desenvolupament del projecte. El conjunt de dades presenta un equilibri acceptable entre les tres classes objectiu: victòria del model A, victòria del model B, i empat, amb distribucions properes al 33% per a cada categoria. No s'han detectat valors nuls en cap de les columnes, simplificant el preprocessament.

Un aspecte crític identificat durant l'exploració és l'existència de 5743 prompts duplicats. Aquesta circumstància representa un risc potencial de data leakage si no es gestiona adequadament, ja que podria inflar artificialment les mètriques de rendiment. Per prevenir-ho, s'ha implementat una estratègia de partició basada en GroupShuffleSplit que garanteix que tots els registres associats al mateix prompt apareguin exclusivament en el conjunt d'entrenament o en el de validació, però mai simultàniament en ambdós.

Respecte a les longituds de les respostes, s'observa una mitjana aproximada de 1550 caràcters per resposta, amb una variància elevada que indica la presència de textos extremadament llargs, arribant en alguns casos a superar els 50000 caràcters. Aquesta distribució presenta reptes importants per als models tipus Transformer, que operen amb límits de tokens fixos.

### Enginyeria de Característiques

S'han desenvolupat diverses categories de característiques numèriques per capturar aspectes estructurals i semàntics de les respostes:

**Característiques de longitud:** Es calculen les longituds del prompt i de cada resposta, així com la diferència absoluta i el ràtio entre les longituds de les dues respostes. Aquestes mètriques s'han revelat com altament predictives, reflectint un biaix humà cap a respostes més extenses que es perceben com més completes.

**Característiques estructurals:** Es quantifiquen elements formals com el nombre d'oracions i la presència de blocs de codi en cada resposta. Aquestes característiques ajuden a identificar respostes ben estructurades i tècnicament elaborades.

**Mesures de similitud:** S'aplica l'índex de Jaccard per quantificar la similitud lèxica entre les dues respostes. Aquesta mètrica s'ha mostrat especialment rellevant per detectar empats, ja que respostes lèxicament similars tendeixen a ser considerades equivalents.

**Característiques del prompt:** Es comptabilitzen elements del prompt com el nombre de preguntes i de missatges, capturant així la complexitat de la tasca plantejada.

Un aspecte fonamental de l'enginyeria de característiques implementada és la decisió deliberada d'excloure les taxes de victòria agregades per model. Tot i que aquestes característiques mostraven correlacions elevades amb el target, s'ha observat que provoquen overfitting, ja que el model aprèn la reputació del generador en lloc de valorar la qualitat real del text. Aquesta estratègia millora la generalització del sistema, especialment davant models no vistos durant l'entrenament.

### Preprocessament Textual

Per a la transformació del text en representacions numèriques aptes per als models de Machine Learning, s'han explorat dues estratègies principals:

**TF-IDF (Term Frequency-Inverse Document Frequency):** Es genera una representació vectorial esparsa que pondera cada paraula segons la seva freqüència al document relativament a la seva freqüència global al corpus. S'ha limitat la dimensionalitat a 5000 característiques i s'han inclòs bi-grams per capturar cert context sintàctic.

**Embeddings Semàntics:** S'utilitza el model all-MiniLM-L6-v2 de SentenceTransformers per generar vectors densos de 384 dimensions que capturen el significat contextual profund del text. Aquesta representació permet reconèixer similaritats semàntiques entre respostes amb formulacions diferents.

Per als models tradicionals, s'ha implementat un procés de normalització mitjançant StandardScaler per a les característiques numèriques, seguit d'una combinació horitzontal amb les representacions textuals. Per al model Transformer, s'ha adoptat una estratègia de Cross-Encoder que concatena el prompt amb ambdues respostes en una única seqüència processada globalment.

### Models Implementats

**Regressió Logística:** Com a baseline sofisticat, s'ha entrenat una regressió logística optimitzada mitjançant GridSearchCV. La millor configuració identificada utilitza el solver newton-cg amb una regularització de C=0.1, reflectint la necessitat de penalitzar fortament la complexitat en presència d'alta dimensionalitat.

**LightGBM amb TF-IDF:** S'implementa un algorisme de Gradient Boosting capaç de capturar relacions no lineals complexes. La configuració inclou learning_rate=0.05, max_depth=7 i num_leaves=31, trobant un equilibri entre capacitat expressiva i control d'overfitting.

**LightGBM amb Embeddings:** Es manté l'arquitectura de boosting substituint TF-IDF per embeddings semàntics. Aquesta variant busca aprofitar la riquesa de les representacions neuronals mantenint l'eficiència dels arbres de decisió.

**DeBERTa-v3-small:** Es desenvolupa una solució basada en Transformers amb arquitectura d'atenció descomposada millorada. S'ha ampliat el límit de processament a 2048 tokens per minimitzar el truncament problemàtic identificat en l'anàlisi exploratòria. L'entrenament s'ha realitzat durant 3 èpoques amb batch size efectiu de 32, learning rate de 2e-5 i weight decay de 0.01.

### Mètrica d'Avaluació

S'ha seleccionat el **Log Loss** com a mètrica principal perquè penalitza severament les prediccions incorrectes realitzades amb alta confiança, forçant el model a expressar adequadament la seva incertesa. Això resulta essencial per obtenir probabilitats ben calibrades que reflecteixin la confiança real del sistema.

Complementàriament, s'utilitzen les **corbes ROC** i l'**àrea sota la corba (AUC)** per avaluar la capacitat discriminativa dels models a través de tots els possibles llindars de decisió, proporcionant una visió de la seva habilitat per separar clarament les diferents classes.

## Resultats

### Comparació de Models

| Model | Log-Loss | Millora sobre Baseline |
|-------|----------|------------------------|
| Baseline (Aleatori) | 1.0986 | 0% |
| Regressió Logística | 1.0570 | 3.8% |
| LightGBM (TF-IDF) | 1.0324 | 6.0% |
| LightGBM (Embeddings) | **1.0301** | **6.2%** |
| DeBERTa-v3 | 1.0707 | 2.5% |

Els resultats obtinguts mostren que els models basats en LightGBM superen clarament les altres aproximacions. La variant que incorpora embeddings semàntics assoleix el millor rendiment amb un Log Loss de 1.0301, seguida molt de prop per la versió basada en TF-IDF amb 1.0324. Aquesta supremacia dels models de boosting confirma la seva capacitat per explotar eficaçment les característiques engineered i capturar relacions no lineals complexes.

La regressió logística obté un rendiment respectable amb 1.0570, demostrant que un model lineal ben regularitzat pot competir quan les característiques estan ben dissenyades. Sorprenentment, el model DeBERTa-v3 presenta un rendiment inferior amb 1.0707, situant-se per sota de la regressió logística.

### Anàlisi de les Matrius de Confusió

Els models LightGBM assoleixen una precisió global del 47%, mostrant un equilibri notable entre les classes "Model A guanya" i "Model B guanya". Les diagonals principals són sòlides, amb aproximadament 2150-2180 encerts per a la classe A i 2015-2024 per a la classe B. Això indica que els models han après eficaçment les regles heurístiques que determinen un guanyador clar.

No obstant això, la detecció d'empats representa el punt feble de tots els models. El LightGBM obté un recall de només 0.35-0.36 per a aquesta classe, identificant correctament uns 1238-1268 empats però confon considerablement empats reals amb victòries aparents. Aquest comportament reflecteix que les característiques estructurals, tot i ser informatives per distingir guanyadors, resulten insuficients per capturar la subtilesa inherent d'un empat.

Curiosament, la regressió logística presenta un recall de 0.42 per a empats, superant el LightGBM en aquest aspecte. Aquest fenomen s'explica per la naturalesa menys agressiva del model lineal: davant del dubte o del soroll en els patrons, tendeix a adoptar una posició més conservadora, cosa que paradoxalment l'ajuda a identificar millor aquells casos on realment no hi ha un guanyador clar.

El model DeBERTa mostra dificultats substancials per discriminar entre les classes A i B, amb una confusió elevada entre elles. Això suggereix que, amb els recursos computacionals disponibles, el model encara no ha aconseguit internalitzar els criteris implícits que determinen per què una resposta és millor que una altra.

### Corbes ROC

L'anàlisi de les corbes ROC proporciona una perspectiva complementària sobre la capacitat discriminativa dels models. Per a la classe "Empat", tots els models presenten valors d'AUC relativament modestos, confirmant la dificultat inherent d'aquesta tasca. Això reflecteix la natura fonamentalment ambigua dels empats: sovint no existeixen característiques objectives que distingeixin clarament un empat genuí d'una victòria ajustada.

Per a les classes "Model A guanya" i "Model B guanya", els models LightGBM aconsegueixen distanciar-se de la regressió logística, exhibint corbes més elevades i valors d'AUC superiors. Això confirma que aquests models han après efectivament a identificar els factors que determinen una victòria clara, aprofitant les característiques estructurals i semàntiques.

## Conclusions i Aprenentatges

Aquest projecte ha proporcionat diverses lliçons fonamentals sobre el desenvolupament de sistemes de Machine Learning per a tasques de Processament de Llenguatge Natural:

**La importància del feature engineering.** La qualitat de les característiques dissenyades pot ser tant o més important que la sofisticació de l'algorisme utilitzat. Les característiques estructurals que capturen diferències de longitud, similitud o elements formals s'han revelat altament predictives de les preferències humanes. Els models que han pogut explotar directament aquestes característiques han aconseguit els millors resultats.

**Embeddings vs. representacions lèxiques.** La incorporació d'embeddings semàntics proporciona millores petites però consistents sobre representacions purament lèxiques com el TF-IDF. Això suggereix que, més enllà de les característiques superficials, existeixen aspectes semàntics profunds que influeixen en la percepció de qualitat, tot i que el seu impacte sigui més subtil.

**Recursos computacionals i Deep Learning.** Els resultats del DeBERTa recorden que els models de Deep Learning, malgrat el seu immens potencial, requereixen recursos substancials de dades, computació i temps per superar enfocaments més tradicionals ben executats. En contextos amb recursos limitats o quan les característiques rellevants poden ser explícitament enginyerades, els mètodes clàssics continuen sent competitius i sovint preferibles per la seva eficiència i interpretabilitat.

**Naturalesa subjectiva de la tasca.** Els resultats relativament modestos en termes absoluts reflecteixen la dificultat inherent de replicar judicis humans sobre preferències textuals. Existeix una considerable subjectivitat en què fa que una resposta sigui millor que una altra, influenciada per factors contextuals, preferències personals sobre l'estil comunicatiu o criteris variables entre avaluadors. Encara que certes preferències sistemàtiques existeixen, no sempre expliquen completament les decisions humanes.

**Gestió de data leakage.** La implementació d'estratègies de partició basades en grups per evitar que el mateix prompt aparegui tant en entrenament com en validació s'ha revelat essencial per obtenir mètriques honestes de generalització.

Aquest treball demostra que la construcció de sistemes efectius de Machine Learning requereix un equilibri cuidadós entre la comprensió del domini, el disseny intel·ligent de característiques, la selecció d'algorismes apropiats i la gestió adequada dels recursos disponibles. Els millors resultats sovint provenen no del model més sofisticat, sinó de l'aplicació reflexiva i ben fonamentada de tècniques adequades al problema específic.

## Estructura del Repositori

```
.
├── Kaggle.ipynb    # Notebook principal amb tot el codi
├── LICENCE         # MIT
└── README.md       # Aquest document
```

## Requisits i Execució

### Llibreries Necessàries

```
pandas
numpy
matplotlib
scikit-learn
scipy
lightgbm
sentence-transformers
transformers
datasets
torch
```

### Execució

Per reproduir els resultats del projecte, executar seqüencialment les cel·les del notebook `Kaggle.ipynb`. 

El model DeBERTa requereix aproximadament 5.5 hores d'entrenament amb un GPU L40S-48Q. El notebook inclou lògica per carregar automàticament el checkpoint preentrenat si existeix a `results_deberta/checkpoint-2600/`, evitant repetir el procés d'entrenament complet.

Els models LightGBM estan configurats per executar-se amb GPU (`device='gpu'`). Si no es disposa de GPU, modificar aquest paràmetre a `device='cpu'`.

## Referències

Aquest projecte s'ha desenvolupat en el context d'una competició de Kaggle sobre predicció de preferències en respostes de models de llenguatge. Les tècniques implementades s'inspiren en metodologies estàndard de Processament de Llenguatge Natural i Machine Learning aplicat a tasques de classificació de text.
