'use client'
import axios from "axios";
import { useState } from "react";


export default function Home() {

  const [newMessages, setNewMessages] = useState('');
  const [messages, setMessages] = useState([{
    sender: 'Model',
    message: 'Hey, how may I help you?'
  }]);
  const [loading, setLoading] = useState(false);

  const handleNewMessage = (event) => {
    setNewMessages(event.target.value);
  }

  const handleMessage = async (event) => {
    const new_Message = {
      sender: 'User',
      message: newMessages
    };

    messages.push(new_Message);
    // setMessages([...messages, new_Message]);
    const question = newMessages;
    setNewMessages('');

    setLoading(true);

    const response = await axios.get(`http://127.0.0.1:5000/ask?query=${question}`);
    console.log(response);
    setNewMessages('');

    setLoading(false);

    messages.push({sender:'Model', message: response.data.answer.result});
    

    console.log(messages);
  }


  return (
    <main className="flex justify-center min-h-screen p-25">
      <div className="w-full font-mono text-sm lg:flex flex-col space-y-8 ">
        
        <div className="flex flex-col space-y-4 mx-10 my-10">
          {
            messages.map((m, index) => (
              <div key={index} className={m.sender=='Model'? "chat chat-start" : "chat chat-end"}>
                <div className="chat-bubble">{m.message}</div>
              </div>
            ))
          }

          <span className={loading? "loading loading-bars loading-md" : "hidden"}></span>

    
        </div>



        <div className="flex items-center justify-around fixed bottom-10 w-full">

          <input type="file" className="file-input file-input-bordered file-input-accent mx-5 min-w-80" />

          <input onChange={handleNewMessage} value={newMessages} type="text" placeholder="Type here" className="input input-bordered w-full mx-5" />

          <button onClick={handleMessage} className="btn btn-outline btn-accent mx-5 px-8"> ASK </button>

        </div>

      </div>
    </main>
  );
}


