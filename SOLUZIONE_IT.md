# La sfida in parole semplici

## 1. Cosa ci chiedono

Dobbiamo scrivere un **programma (un "agente")** che gestisce da solo una piccola
azienda di elettronica, in gara contro le aziende delle altre squadre.

Ogni turno l'agente deve decidere tre cose:

1. **Cosa comprare** — materie prime (CHIP, DISPLAY, BATTERY, CASING) all'asta.
2. **Cosa produrre** — con le materie prime monta prodotti (PHONE, LAPTOP,
   TABLET, SMARTWATCH, CONSOLE), seguendo delle "ricette".
3. **A che prezzo vendere** — mette i prodotti in vendita e arrivano i clienti.

**Vince chi finisce la partita con più soldi in cassa.** I prodotti rimasti in
magazzino alla fine valgono **zero**.

La classifica finale è la **media delle posizioni** in tante partite giocate su
**scenari segreti** (mercati diversi da quelli che vediamo). Quindi non basta
essere bravi in un mercato: bisogna adattarsi a qualunque mercato.

## 2. Come funziona un turno

Tutte le decisioni del turno si prendono **insieme, all'inizio**, senza sapere
cosa fanno gli altri. Poi il simulatore esegue tre fasi:

| Fase              | Cosa succede                                                                                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Asta**       | Ogni materia prima ha un lotto con un **prezzo minimo**. Le offerte sono segrete; vince chi offre di più e **paga quello che ha offerto** (se offri 100 paghi 100, anche se bastava 30). |
| **2. Produzione** | I materiali comprati si possono usare **subito**, nello stesso turno.                                                                                                                    |
| **3. Vendita**    | Arrivano i clienti. Ognuno vuole **un** prodotto, ha un **prezzo massimo** che è disposto a pagare, e compra **una** delle offerte che costano meno di quel massimo.                     |

Altre regole utili:

- **Niente costi di magazzino**: quello che non vendi resta per il turno dopo, gratis.
- Vediamo il magazzino e i soldi di **tutte** le squadre, ma **non** i loro prezzi.
- Vediamo solo **le nostre** vendite e i nostri acquisti passati.
- **Non sappiamo quanti turni dura la partita** (nell'esempio sono 40).
- L'agente ha **0,1 secondi** per decidere ogni turno.

## 3. Le due cose che contano davvero

Leggendo il codice del simulatore abbiamo scoperto due dettagli decisivi:

**a) I materiali costano poco, i clienti pagano tanto.**
Un LAPTOP costa circa 70 di materiali, ma un cliente "premium" lo paga anche
800–1400. Quindi il vero problema **non** è risparmiare sui materiali:
è **non perdere clienti** (avere sempre merce) e **scegliere bene il prezzo**.

**b) Come sceglie il cliente.**
Il cliente guarda tutte le offerte che può permettersi e ne sceglie una **a
caso**, ma con più probabilità quelle più economiche. Ogni **offerta** conta
come un "biglietto della lotteria". Se mettiamo in vendita 10 pezzi come **10
offerte separate** invece di una sola offerta da 10, abbiamo più biglietti e
quindi più clienti. Le regole lo permettono esplicitamente ("puoi dividere un
prodotto in più offerte a prezzi diversi").

Inoltre, mettendo i pezzi a **prezzi diversi** (una "scala" di prezzi), i
clienti ricchi a volte comprano il pezzo caro: guadagniamo di più senza perdere
quelli che hanno meno soldi.

## 4. La nostra soluzione

Abbiamo già un progetto (`bizagent`) scritto prima di conoscere le regole. Ora
che le regole sono note, costruiamo un agente **su misura** per questo gioco
(`rotating_forks_agent`), riusando le idee migliori di `bizagent`.

Ogni turno l'agente fa così:

1. **Prezzi che imparano da soli.** Per ogni prodotto c'è un "prezzo di
   riferimento". Si parte alto (8 volte il costo dei materiali). Se vendiamo un
   pezzo al prezzo di riferimento o sopra, lo alziamo del 5%; se non vendiamo
   niente, lo abbassiamo del 10%.
2. **Scala di prezzi.** I pezzi vengono messi in vendita a prezzi un po' diversi
   intorno al riferimento (da -20% a +40%), così i clienti ricchi a volte
   comprano quello caro.
3. **Un'offerta per ogni pezzo**, così abbiamo più "biglietti" per i clienti.
4. **Pavimento di prezzo.** Non vendiamo mai sotto **2,5 volte il costo**. Questo
   evita la "guerra dei prezzi" quando ci sono tante squadre (è stata la
   scoperta più importante dei test).
5. **Pezzi "sonda".** Se un prodotto non ha **mai** venduto, mettiamo anche un paio
   di pezzi molto scontati (60% e 40% del riferimento) per capire subito quanto
   pagano i clienti. Se perfino il pavimento è troppo alto per quel mercato,
   l'agente lo abbassa da solo.
6. **Scaffale pieno.** Quando un prodotto inizia a vendere, teniamo circa **14
   pezzi** pronti (più quanto vendiamo di solito in 2 turni): più pezzi in
   vendita = più clienti. All'inizio, finché non vende, ne teniamo solo 3.
7. **Produzione intelligente.** Il CHIP serve a tutti i prodotti: lo diamo prima ai
   prodotti che rendono di più per ogni unità di materiale.
8. **Asta prudente.** Offriamo il 10% sopra il prezzo minimo. Se non riusciamo a
   comprare (altre squadre offrono di più), alziamo l'offerta, fino a un massimo.
   Non spendiamo mai più del 70% della cassa in un turno.
9. **Sicurezza.** Se qualcosa va storto nel codice, l'agente non perde il turno:
   mette comunque in vendita quello che ha in magazzino.

**Fine partita:** non sappiamo quanti turni dura, quindi non svuotiamo il
magazzino "in anticipo" (lo abbiamo provato: rischia di perdere vendite). Il
costo dei pezzi avanzati è piccolo rispetto ai guadagni.

Come **paracadute** c'è anche un agente semplice (`rotating_forks_baseline`).

## 5. Come lo verifichiamo

- `tools/league.py` gioca tante partite e calcola la **posizione media**, cioè la
  stessa metrica della classifica ufficiale.
- `tools/evolve.py` cerca automaticamente i parametri migliori: 8 versioni
  dell'agente giocano nello stesso mercato, le migliori sopravvivono.
- Visto che gli scenari ufficiali sono segreti, abbiamo creato **8 mercati di
  prova** in `tools/scenarios/` (clienti più o meno sensibili al prezzo,
  materiali scarsi o costosi, produzione limitata, partite corte/lunghe, clienti
  più poveri), oltre ai 4 mercati ufficiali.

## 6. Come si usa (per chi lavora sul codice)

```bash
cd pmds-pricing-hackathon
uv sync --locked --python 3.12
uv run dme-sim --turns 40 --ui none          # una partita
uv run dme-sim --turns 40 --ui window        # con la finestra grafica
```

Gli agenti stanno in `src/dme_simulator/agents/<nome>/agent.py`. Per la
consegna si carica sul portale **solo la cartella dell'agente**.

Per provare il nostro agente in un mercato affollato:

```bash
uv run python tools/league.py --roster crowded --seeds 10
```

## 7. Risultati

12 mercati × 10 partite = **120 partite** per ogni prova (file completi in
`tools/results/`). La posizione media migliore possibile è 1,00.

**Contro i 3 bot ufficiali** (Dummy, Patient, PriceWar):

| Agente | Posizione media | Vittorie | Cassa media |
|---|---|---|---|
| **rotating_forks_agent** | **1,00** | **120 / 120** | 140.973 |
| DummyAgent | 2,04 | 0 | 18.644 |
| PatientAgent | 3,33 | 0 | 14.250 |
| PriceWarAgent | 3,62 | 0 | 11.911 |

**In un mercato affollato** (8 squadre, tra cui le nostre versioni precedenti):

| Agente | Posizione media | Vittorie | Cassa media |
|---|---|---|---|
| **rotating_forks_agent** (finale) | **1,07** | **112 / 120** | 60.500 |
| pepo_v1 (prima versione) | 2,29 | 7 | 28.563 |
| pepo_single (senza offerte separate) | 2,89 | 1 | 25.092 |
| rotating_forks_baseline / copia | ~4,6 | 0 | ~14.700 |
| DummyAgent | 5,78 | 0 | 12.499 |
| PriceWarAgent (×2) | ~7,3 | 0 | ~10.600 |

Il nostro agente arriva primo in media in tutti i 12 mercati. I punti un po' più
deboli sono "materiali costosi" (posizione media 1,70) e "materiali scarsi"
(1,10).

Tempo di decisione: circa **0,2 millisecondi** per turno (il limite è 100).

**Attenzione — cosa questi numeri NON dicono:** gli avversari veri sono le
altre squadre, non i nostri bot. Agenti molto diversi dai nostri potrebbero
comportarsi in modo che non abbiamo previsto. Inoltre il repository è pubblico:
le altre squadre possono vedere il nostro codice.

## 8. Consegna

Il file pronto da caricare è **`submission/rotating_forks_agent.zip`** (contiene solo
`rotating_forks_agent/agent.py`, 6 KB). Lo abbiamo già verificato come farebbe il portale:
compila, si carica da una cartella separata e gioca senza errori.

Paracadute: se per qualche motivo il portale lo rifiuta, caricare la cartella
`src/dme_simulator/agents/rotating_forks_baseline/`.

## 9. Dove sta cosa

| Percorso | Cosa contiene |
|---|---|
| `src/dme_simulator/agents/rotating_forks_agent/` | **Il nostro agente** (quello da consegnare) |
| `src/dme_simulator/agents/rotating_forks_baseline/` | Agente semplice di riserva |
| `submission/rotating_forks_agent.zip` | File pronto da caricare sul portale |
| `tools/` | Test: `league.py` (classifica), `evolve.py` (ricerca parametri), `scenarios/` (mercati di prova), `results/` |
| `legacy/bizagent/` | Il vecchio progetto scritto prima di conoscere le regole. **Non serve per la gara**, lo teniamo solo come riferimento |
| tutto il resto | Il simulatore ufficiale degli organizzatori (da non modificare) |

## 10. Versione 2 (dopo la prima classifica)

**Cosa abbiamo scoperto guardando la gara vera.** Nelle fork pubbliche solo 4
squadre hanno messo il codice: quasi tutte usano l'agente di esempio del README
(compra 1 pezzo di ogni materiale a prezzo minimo +10%, costruisce 1 PHONE,
vende tutto a 250). I primi in classifica sono privati.

Con 25 squadre la partita è molto diversa dai nostri primi test:

1. **All'inizio i materiali finiscono subito.** ~20 squadre offrono il prezzo
   minimo +10%, cioè **esattamente quanto offrivamo noi**: a parità di prezzo
   l'asta si divide a turno e noi non compravamo quasi niente per i primi turni.
   Dal turno ~10 i fornitori producono di più e i materiali tornano abbondanti.
2. **Ognuno vende pochi pezzi** (i ~25 clienti per turno si dividono tra tante
   squadre). Il nostro scaffale da 14 pezzi per prodotto faceva finire la
   partita con decine di pezzi invenduti, che valgono zero.

**Cosa abbiamo cambiato:**

| Prima | Ora |
|---|---|
| Asta: prezzo minimo +10% dal primo turno | **Prezzo minimo ×2 al primo turno**, poi scende da solo quando i lotti diventano abbondanti (sale fino a ×4 se perdiamo) |
| Scaffale che cresce fino a ~14 pezzi per prodotto | **Scaffale = 2 pezzi + 3 turni di vendite reali**: grande dove vendiamo, piccolo dove no |
| Fino al 70% della cassa impegnabile in un turno | **Massimo 50%** |

**Come l'abbiamo verificato.** `tools/opponents/` contiene due ricostruzioni della
gara: `real_field.json` (come la prima classifica) e `tough_field.json` (come
la seconda, con 5 avversari "furbi" al posto di 5 agenti di esempio). Lo script
`tools/variants.py` cambia **solo** il nostro agente e lascia uguale il resto.

Risultati su semi nuovi (posizione media, più basso è meglio, 7 mercati × 12 partite):

| Campo | Versione 2 | Versione 1 (quella caricata prima) |
|---|---|---|
| Campo "duro" (come la gara di adesso, 24 squadre) | **6,38** | 11,07 |
| Campo come la prima classifica (25 squadre) | **1,74** | 5,29 |
| Contro i 3 bot ufficiali (12 mercati) | **1,00** (120/120) | 1,00 |

Punto ancora debole: partite corte (15 turni) in un mercato affollato.

Per rifare i test servono anche gli agenti delle altre squadre (non li
copiamo nel nostro repo): si scaricano dalle loro fork e si passano con
`--extra-dirs <cartella>`.

## 11. Versione 3 — attaccare all'inizio e leggere gli avversari

L'agente ora si chiama **`rotating_forks_agent`** (file da caricare:
`submission/rotating_forks_agent.zip`).

**Cosa ci hanno insegnato i grafici della gara.** Il primo in classifica guadagna
quasi tutto nei **primi 5 turni** e poi resta fermo. Nei primi turni i materiali
scarseggiano e quasi nessuno ha LAPTOP o CONSOLE: chi li ha subito, a prezzi
alti, prende tutti i clienti "premium" (che pagano un LAPTOP fino a 1.400).
Noi eravamo troppo **passivi** proprio lì.

**Cosa abbiamo cambiato:**

| Cosa | Prima (v2) | Ora (v3) |
|---|---|---|
| Offerta all'asta al turno 1 | 2× il prezzo minimo | **3×** il prezzo minimo |
| Pezzi pronti prima di vendere | 6 per prodotto | **10** per prodotto |
| Prezzo di partenza | 8× il costo | **10×** il costo (approfittiamo del monopolio iniziale) |
| Scaffale dopo le prime vendite | 2 + 3 turni di vendite | 2 + **5** turni di vendite |

**Lettura degli avversari** (regole che si vedono solo leggendo il codice):

- **Magazzini di tutti visibili** (`public_inventories`): per ogni prodotto
  contiamo quanti concorrenti lo hanno. Se **nessuno** lo ha, siamo in monopolio
  e alziamo il prezzo più in fretta. I materiali scarsi vanno prima ai prodotti
  con meno concorrenti.
- **Vendite degli altri**: quando il magazzino di un concorrente cala, ha
  venduto. Se gli altri vendono un prodotto e noi no, siamo troppo cari:
  scendiamo più in fretta.
- **Lotti dell'asta**: se il magazzino del fornitore cresce, c'è più offerta che
  domanda: offriamo il prezzo minimo senza pagare extra.
- **Regola nascosta `observe_all_transactions`**: uno scenario segreto può
  mostrare a tutti **tutte** le transazioni (con prezzi e venditori). Un agente
  che conta tutte le vendite come proprie (come faceva il nostro, e
  probabilmente molti altri) si confonde. Ora l'agente **capisce qual è la sua
  squadra** (cassa e magazzino uguali ai suoi, o prezzi identici ai suoi) e usa
  le vendite degli altri come informazione: se loro vendono e noi no, va subito
  al prezzo più alto a cui il mercato sta comprando.

**Risultati su semi nuovi** (posizione media, 9 mercati × 12 partite):

| Campo | v3 | v2 |
|---|---|---|
| Campo "duro" (24 squadre) | **2,99** (78 vittorie su 108) | 4,45 (36 vittorie) |
| Campo come la prima classifica | **1,34** (82 vittorie su 108) | 1,71 (45 vittorie) |
| Contro i 3 bot ufficiali (14 mercati) | **1,00** (112/112) | 1,00 |

Punti ancora deboli nel campo duro: materiali costosi (posizione ~9) e clienti
poveri (~6,5).

**Cosa abbiamo provato e scartato:** fermarsi a un turno fisso per "tenersi il
guadagno" (peggiora: nei nostri test continuare a vendere conviene ancora).
