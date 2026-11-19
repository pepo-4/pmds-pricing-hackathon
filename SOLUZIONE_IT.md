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
(`444_agent`), riusando le idee migliori di `bizagent`.

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

Come **paracadute** c'è anche un agente semplice (`pepo_baseline`).

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
| **444_agent** | **1,00** | **120 / 120** | 140.973 |
| DummyAgent | 2,04 | 0 | 18.644 |
| PatientAgent | 3,33 | 0 | 14.250 |
| PriceWarAgent | 3,62 | 0 | 11.911 |

**In un mercato affollato** (8 squadre, tra cui le nostre versioni precedenti):

| Agente | Posizione media | Vittorie | Cassa media |
|---|---|---|---|
| **444_agent** (finale) | **1,07** | **112 / 120** | 60.500 |
| pepo_v1 (prima versione) | 2,29 | 7 | 28.563 |
| pepo_single (senza offerte separate) | 2,89 | 1 | 25.092 |
| pepo_baseline / copia | ~4,6 | 0 | ~14.700 |
| DummyAgent | 5,78 | 0 | 12.499 |
| PriceWarAgent (×2) | ~7,3 | 0 | ~10.600 |

Il nostro agente arriva primo in media in tutti i 12 mercati. I punti un po' più
deboli sono "materiali costosi" (posizione media 1,70) e "materiali scarsi"
(1,10).

Tempo di decisione: circa **0,2 millisecondi** per turno (il limite è 100).

**Aggiornamento competitivo finale.** Il modello ora considera anche lo stock
pubblico dei prodotti e la pressione sulle materie prime. Su una verifica nuova
di 96 partite contro i bot ufficiali distribuite sui 12 mercati disponibili,
`444_agent` ha ottenuto posizione media 1,00 su tutti i 96. Su 96 partite in
mercato affollato, ha ottenuto posizione media **1,05**, con 91 vittorie. Nel
mercato con materiali costosi ha pareggiato la prima posizione con `pepo_v1`
(1,50 ciascuno), migliorando nettamente il punto più debole della versione
precedente.

**Contromossa contro bot speculari.** La lega speculare include sei agenti
plausibili sviluppati nello stesso stile:

- ladder bilanciata, conservativa e aggressiva;
- stock ottimizzato per mercati brevi;
- stock ottimizzato per mercati lunghi;
- sniper d'asta e ladder.

Su 60 partite tra i mirror più rappresentativi, il profilo `counter_sniper`
ha ottenuto posizione media **1,35** con 39 vittorie, contro 2,22 e 16 vittorie
per `mirror_sniper`. Il default finale combina stock compatto, ladder ampia,
offerta d'asta al 45% sopra riserva e budget d'acquisto del 90%. Questo profilo
ha mantenuto posizione media 1,00 contro 60 test ufficiali su mercati brevi,
lunghi, standard, costosi e scarsi, oltre a vincere nei roster aggressivi e
passivi testati.

**Adattamento a mercati da 15, 80 e 240 turni.** Sono stati aggiunti scenari
ultralunghi e sei bot con strategie non copiate dalla baseline:

- undercut continuo;
- accumulo aggressivo di scorte;
- controllo del CHIP;
- conservazione di liquidità;
- adattamento ai capitali pubblici;
- profilo maturo oltre il turno 80.

L'agente rileva quattro regimi osservabili: mercato costoso, mercato con offerta
scarsa, mercato maturo oltre il turno 80 e mercato normale. Nei mercati costosi
abbassa markup iniziale e pavimento, usa una ladder sotto-costo e riduce lo
stock; nei mercati scarsamente riforniti mantiene copertura moderata; oltre il
turno 80 usa scorte alte, ladder stabile e aste moderate.

In 20 partite da 240 turni contro stockpile e bot speculari, `444_agent` ha
ottenuto 11 vittorie e posizione media 1,45, contro stockpile 9 vittorie e
posizione media 1,55. Contro i bot ufficiali ha vinto 48/48 su ultralong,
costoso, scarso e capacità.

**Aggiornamento basato su Tight v7.** Il bot vincente usa quattro leve che
mancavano alla nostra implementazione precedente:

- **stock dinamico per vendite previste:** copertura di circa 3 turni, non un
  target fisso.
- **bid cap adattivo:** cresce solo quando abbiamo perso materiali all'asta e
  vendiamo quasi tutto; si restringe quando lo stock finito accumula.
- **cap per valore:** nessun materiale viene pagato oltre una quota del valore
  dei prodotti migliori che può creare.
- **lettura pubblica della concorrenza:** i beni non offerti dai rivali ricevono
  una ladder più alta e i materiali scarsi vengono assegnati ai beni meno
  presenti nei mercati avversari.

Il modello attuale usa un regime aggressivo nei mercati normali, lunghi e con
offerta scarsa, e passa a un profilo conservativo nei mercati con riserve
materiali costose. Su 120 partite tra mercati standard, brevi, lunghi, costosi,
scarsi e con capacità limitata, `444_agent` ha ottenuto posizione media **2,38**
e 57 vittorie. A 240 turni ha ottenuto 7 vittorie su 12 e posizione media
**1,42**, superando Tight aggressivo con 5 vittorie e posizione media 1,75.

**Ottimizzazione dei mercati da pochi turni.** Sono state confrontate 12
versioni Tight su mercati da 15 turni e sui tutorial. Nei mercati normali il
profilo migliore disattiva l'apertura aggressiva, tiene stock minimo 1,
copertura di 1.5 turni, massimo 24 pezzi finiti e bid cap 4x riserva.

Su 20 partite da 15 turni:

- `444_agent`: posizione media **1,15**, 18 vittorie
- `tight`: posizione media 3,20
- `tight_conservative`: 3,60
- `tight_v6`: 3,85

**Limite dell'aggressione estrema.** Nelle due configurazioni principali da
40 e 80 turni abbiamo testato profili all-in con markup 2,5–2,8x, stock fino
a 60 unità e aste fino al 100% della cassa. Questi profili hanno rapidamente
collassato sul capitale. Il modello vincente usa aggressione moderata:
markup iniziale normalizzato nel mercato, stock controllato e reinvestimento
limitato all'80% nei mercati costosi. Su 60 partite da 40/80/costoso il profilo
base ha vinto 40/60; nel costoso il contesto conserva invece il vantaggio agli
agenti liquidi e undercutter.

**Attenzione — cosa questi numeri NON dicono:** gli avversari veri sono le
altre squadre, non i nostri bot. Agenti molto diversi dai nostri potrebbero
comportarsi in modo che non abbiamo previsto. Inoltre il repository è pubblico:
le altre squadre possono vedere il nostro codice.

## 8. Consegna

Il file pronto da caricare è **`submission/444_agent.zip`** (contiene solo
`444_agent/agent.py`, 4 KB). Lo abbiamo già verificato come farebbe il portale:
compila, si carica da una cartella separata e gioca senza errori.

Paracadute: se per qualche motivo il portale lo rifiuta, caricare la cartella
`src/dme_simulator/agents/pepo_baseline/`.

## 9. Dove sta cosa

| Percorso | Cosa contiene |
|---|---|
| `src/dme_simulator/agents/444_agent/` | **Il nostro agente** (quello da consegnare) |
| `src/dme_simulator/agents/pepo_baseline/` | Agente semplice di riserva |
| `submission/444_agent.zip` | File pronto da caricare sul portale |
| `tools/` | Test: `league.py` (classifica), `evolve.py` (ricerca parametri), `scenarios/` (mercati di prova), `results/` |
| `legacy/bizagent/` | Il vecchio progetto scritto prima di conoscere le regole. **Non serve per la gara**, lo teniamo solo come riferimento |
| tutto il resto | Il simulatore ufficiale degli organizzatori (da non modificare) |
