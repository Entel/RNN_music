import tensorflow as tf
import midi
import numpy
import os

lowerBound = 24
upperBound = 102

# midi to note
def midiToNoteStateMatrix(midifile):

    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    span = upperBound-lowerBound
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)
    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0]
                        else:
                            state[evt.pitch-lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        return statematrix

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    return statematrix

def noteStateMatrixToMidi(statematrix, name="output_mid"):
    statematrix = numpy.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]): 
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern) 

def get_songs(midi_path):
    files = os.listdir(midi_path)
    songs = []
    for f in files:
        f = midi_path + '/' + f
        try:
            song = numpy.array(midiToNoteStateMatrix(f))
            # print song.shape
            if numpy.array(song).shape[0] > 64:
                songs.append(song)
                print('loaded:', f)
        except Exception as e:
            print('Error:', e)
    print('Finish loading ' + str(len(songs)) + ' songs.')
    return songs

songs = get_songs('midi')

note_range = upperBound - lowerBound

n_timesteps = 128
n_input = 2 * note_range * n_timesteps
n_hidden = 64

X = tf.placeholder(tf.float32, [None, n_input])
W = None
bh = None
bv = None

def sample(probs):
    return tf.floor(probs + tf.random_uniform(tf.shape(probs),0, 1))

def gibbs_sample(k):
    def body(count, k, xk):
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))
        return count+1, k, xk

    count = tf.constant(0)
    def condition(count, k, xk):
        return count < k
    [_, _, x_sample] = tf.while_loop(condition,body, [count, tf.constant(k), X])
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

def neural_network():
    global W, bh, bv
    W = tf.Variable(tf.random_normal([n_input, n_hidden], 0.01))
    bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32))
    bv = tf.Variable(tf.zeros([1, n_input], tf.float32))

    x_sample = gibbs_sample(1)
    h = sample(tf.sigmoid(tf.matmul(X, W)+ bh))
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

    learning_rate = tf.constant(0.005, tf.float32)
    size_bt = tf.cast(tf.shape(X)[0], tf.float32)
    W_adder = tf.mul(learning_rate/size_bt, 
        tf.sub(tf.matmul(tf.transpose(X), h), 
        tf.matmul(tf.transpose(x_sample), h_sample)))
    bv_adder = tf.mul(learning_rate/size_bt, tf.reduce_sum(tf.sub(X, x_sample), 0, True))
    bh_adder = tf.mul(learning_rate/size_bt, tf.reduce_sum(tf.sub(h, h_sample), 0, True))
    update = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]
    return update

def train_neural_network():
    update = neural_network()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        saver = tf.train.Saver(tf.all_variables())
        
        epochs = 2048 * 1024
        batch_size = 64
        for epoch in range(epochs):
            for song in songs:
                song = numpy.array(song)
                song = song[:int(numpy.floor(song.shape[0]/n_timesteps) * n_timesteps)]
                song = numpy.reshape(song, [song.shape[0]/n_timesteps, song.shape[1] * n_timesteps * 2])
            
            for i in range(1, len(song) * 4, batch_size):
                train_x = song[i:i+batch_size]
                sess.run(update, feed_dict={X: train_x})
                
            print(epoch)
            if epoch == epochs - 1:
                saver.save(sess, 'midi.module')

        sample = gibbs_sample(1).eval(session = sess, feed_dict = {X: numpy.zeros((1, n_input))})
        print (n_timesteps, 2 * note_range)
        S = numpy.reshape(sample[0, :], (n_timesteps, note_range, 2))
        print 's', S.shape
        noteStateMatrixToMidi(S, 'output')
        print('Builded output.mid!')

train_neural_network()
# noteStateMatrixToMidi(midiToNoteStateMatrix('../midi/beethoven_opus10_2.mid'))

