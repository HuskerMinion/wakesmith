# wakesmith auf Deutsch (experimentell)

wakesmith kann testweise deutsche Wake Words trainieren, zum Beispiel „hallo computer“, für
ESPHome-Geräte (microWakeWord, z. B. Home Assistant Voice PE) und für openWakeWord. Das ist neu und
noch nicht ausgereift. Danke, dass du es ausprobierst! Am Ende steht, welche Rückmeldung am meisten
hilft.

Die Oberfläche ist auf Englisch. Die Namen der Schaltflächen stehen hier **fett**, so wie sie auf der
Seite heißen.

## Was du brauchst

- **System:** Linux (z. B. Ubuntu) oder Windows 11 mit WSL2. macOS ist noch nicht getestet und
  rechnet dort nur auf dem Prozessor.
- **Grafikkarte:** NVIDIA, Intel Arc oder AMD (experimentell). Ohne GPU geht es auch, nur langsamer.
- **Speicher:** etwa 25 GB freier Plattenplatz, 16 GB RAM oder mehr
- **Mikrofon** für echte Aufnahmen im Browser. Deine Stimme und die deiner Familie machen das Modell
  viel besser.

## 1. Installieren

Lade die neueste `wakesmith-<version>-py3-none-any.whl` (0.4.0 oder neuer) von der
[Releases-Seite](https://github.com/HuskerMinion/wakesmith/releases) herunter. Dann in einem
Terminal, im Ordner mit der Datei:

```bash
sudo apt install espeak-ng git                       # Debian/Ubuntu; espeak-ng wandelt Text in Lautschrift um
# auf macOS stattdessen: brew install espeak-ng git
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv installiert Python-Programme
uv tool install --python 3.12 ./wakesmith-<version>-py3-none-any.whl
wakesmith setup      # passendes PyTorch für deine GPU, lädt ~24 GB Daten
wakesmith serve      # dann http://localhost:8765/ im Browser öffnen
```

wakesmith braucht Python 3.10 bis 3.12; `--python 3.12` sorgt dafür, dass uv die passende Version
nimmt (und sie bei Bedarf lädt). Auf macOS statt apt: `brew install espeak-ng git`.

Nach der uv-Installation einmal ein neues Terminal öffnen, damit der Befehl `wakesmith` gefunden
wird. `wakesmith setup` dauert je nach Leitung eine Weile. `wakesmith doctor` zeigt, ob alles da ist
und welche GPU erkannt wurde.

## 2. Deutsch einschalten

Öffne links **Data & licenses** und setze im Abschnitt **Languages** den Haken bei **Offer German for
new wake words (experimental)**.

Auf der Kommandozeile geht es so:

```bash
WAKESMITH_EXPERIMENTAL_LANGUAGES=1 wakesmith new "hallo computer" --language de
```

## 3. Das Wort anlegen

- Links **New wake word**, dann den Einsatzzweck wählen (z. B. ESPHome-Gerät).
- Bei **Language** **Deutsch (experimental)** wählen und die Phrase eintippen, klein geschrieben. Umlaute sind
  erlaubt.
- **Hear it** spielt die Phrase mit einer deutschen Stimme ab. Klingt sie falsch, schreib sie so, wie
  man sie spricht.
- **Suggest sound-alikes** schlägt ähnlich klingende Wörter vor, auf die das Modell *nicht* reagieren
  soll. Unpassende einfach wegklicken, eigene ergänzen.

Gute Wake Words haben drei bis fünf Silben und kommen im Alltag selten genau so vor. „Computer“
allein löst öfter aus als „hallo computer“.

Für den ersten Durchlauf unter **Advanced** die Größe **Quick test** wählen: Dann siehst du in wenigen
Minuten, ob alles läuft. Für ein brauchbares Modell danach **Standard**.

## 4. Echte Aufnahmen machen

Mit **Next: real voices** geht es zum Aufnehmen. Unter **Record here** auf das Mikrofon drücken und die
Phrase einmal sagen. Die Aufnahme endet von selbst. Die Seite sagt dir jedes Mal, wie du sie sagen
sollst (leise, laut, von weiter weg, mit Fernseher im Hintergrund).

- Ziel: mindestens 40 bis 50 Aufnahmen, gern von mehreren Personen. Trag bei jeder Aufnahme den Namen
  der Person ein.
- Jede vierte Aufnahme wird zurückgelegt und nie trainiert. Mit diesen Aufnahmen wird das Modell
  ehrlich getestet.
- Ein paar Aufnahmen mit Raumgeräusch (Gespräche, Fernseher, ohne das Wort) helfen gegen
  Fehlauslösungen.

> **Auf Deutsch ist das wichtiger als auf Englisch.** Für Englisch gibt es über 900 synthetische
> Sprecher, für Deutsch nur eine Handvoll Stimmen. In unserem Test hat ein Modell nur aus
> synthetischen Stimmen eine deutsche Stimme, die es nie gehört hatte, meist erkannt, Sprecher mit
> fremdem Akzent aber kaum. Echte Aufnahmen aus deinem Haushalt sind der größte Gewinn.

Das Mikrofon funktioniert nur auf `localhost` oder über HTTPS. Läuft wakesmith auf einem anderen
Rechner, starte es mit `wakesmith serve --host 0.0.0.0 --https` und bestätige die Zertifikatswarnung
einmal. Der Tab **Common Voice** hilft bei Deutsch noch nicht.

## 5. Trainieren

**Train** startet das Training. Fortschritt und Restzeit stehen auf der Seite **Training**.

Beim ersten deutschen Wort werden die deutschen Stimmen geladen (etwa 0,5 GB). Bei der Größe
**Standard** dauert ein Durchlauf mit einer GPU etwa eine halbe Stunde (gemessen auf einer RTX 5060 Ti),
ein **Quick test** etwa 10 Minuten; ohne GPU deutlich länger.

## 6. Ergebnis prüfen und exportieren

Unter **Results & export** siehst du, wie viele deiner zurückgelegten Aufnahmen erkannt werden und wie
oft das Modell pro Stunde fälschlich auslöst. Dort wählst du die Schwelle (**cutoff**) und lädst das
Modell herunter:

- **microWakeWord:** `.tflite` und `.json` plus ein ESPHome-YAML-Schnipsel
- **openWakeWord:** `.tflite` für das Wyoming-Add-on

Das Modell meldet seine Sprache als `trained_languages: ["de"]`.

## Was am meisten hilft

Bitte eröffne ein [Issue auf GitHub](https://github.com/HuskerMinion/wakesmith/issues), gern auf
Deutsch, mit:

- Phrase, Gerät und GPU (die Ausgabe von `wakesmith doctor` genügt)
- Wie viele Aufnahmen von wie vielen Personen
- Im Alltag: Wie oft reagiert das Gerät bei 20 Versuchen? Und wie oft löst es pro Tag ungewollt aus?
- Klingen die deutschen Stimmen bei **Hear it** richtig? Passen die vorgeschlagenen Sound-alikes?
- Bei Fehlern: die letzten Zeilen aus dem Protokoll auf der Seite **Training**

## Lizenzen

wakesmith ist für nichtkommerzielle Nutzung frei ([PolyForm Noncommercial](../LICENSE)). Von den
deutschen Stimmen sind Thorsten und Kerstin CC0, Pavoque ist CC BY-NC-SA (nicht kommerziell). Eva K,
Karlsson und Ramona stammen aus dem M-AILABS-Datensatz, dessen Bedingungen Piper nicht ausdrücklich
nennt; wakesmith behandelt sie deshalb als nur privat nutzbar. Das gilt damit auch für Modelle, die
mit ihnen trainiert wurden.
