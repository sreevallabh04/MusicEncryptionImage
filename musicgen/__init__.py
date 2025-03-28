from typing import List, Tuple, Union

from music21 import converter
from music21.key import Key
from music21.note import Note
from music21.stream import Stream

from musicgen.chordcreator import ChordCreator
from musicgen.rules import Rules, TriadBaroque, TriadBaroqueCypher, Cypher

NoteIdentifier = Union[Tuple[str, Union[float, int]], Tuple[str, Union[float, int], float]]


def create_chords(notes_in: List[NoteIdentifier], ruleset: Rules = TriadBaroque()) -> Stream:
    """
    Creates the Stream of Chords made with the input notes. Notes are represented as (name, quarterLength) or
    (name, quarterLength, volumes) pairs.

    By default, the key is guessed.
    :param notes_in: A list of note identifiers that will be converted into
    :param ruleset: A Rules object that determines how the Chords are fitted. Defaults to a Rules object that generates
    triads based on the rules from the Baroque period.
    :return: A Stream containing the generated Chords.
    """
    notes_out: List[Note] = []
    for note in notes_in:
        note_out = Note(note[0])
        note_out.quarterLength = note[1]
        if len(note) == 3:
            note_out.volume = note[2]

        notes_out.append(note_out)

    chord_creator = ChordCreator(notes_out)

    return chord_creator.chordify(ruleset)


def decode(music_in: str, cypher: Cypher) -> List[NoteIdentifier]:
    """
    Extracts from a *.md file the notes and it's associated information and creates a list
    :param music_in: the music file to extract the infromation from
    :param cypher: the cypher to use to decode the file
    :return: a list of associated note, quarter_length and volume
    """
    out: List[NoteIdentifier] = []
    stream: Stream = converter.parse(music_in)
    notes = cypher.decode(stream.flatten())
    # print(notes)
    # stream.show()

    for note in notes:
        out.append((note.name, float(note.quarterLength), float(note.volume.velocity)))

    return out


if __name__ == '__main__':
    test_cypher = TriadBaroqueCypher(Key('a'))
    test = create_chords([("B", 1, 10), ("F", 1), ("A", 1), ("G#", 1), ("D", 1, 10), ("C", 1.0), ("B", 1), ("E", 1)],
                         TriadBaroqueCypher(Key('a')))
    print(test_cypher.decode(test))
