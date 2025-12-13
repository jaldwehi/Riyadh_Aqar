const API_BASE = "http://127.0.0.1:8000";
const PREDICT_URL = `${API_BASE}/predict_price`;
const PING_URL = `${API_BASE}/`;

const el = (id) => document.getElementById(id);

/* RANDOM HOUSE IMAGES */
const HOUSE_IMAGES = [
  "./assets/houses/h1.jpg",
  "./assets/houses/h2.jpg",
  "./assets/houses/h3.jpg",
  "./assets/houses/h4.jpg",
  "./assets/houses/h5.jpg",
  "./assets/houses/h6.jpg",
];

const pickHouseImage = () =>
  HOUSE_IMAGES[Math.floor(Math.random() * HOUSE_IMAGES.length)];

function formatSAR(n){
  return `${Math.round(n).toLocaleString("en-SA")} SAR`;
}

function readInputs(){
  return {
    neighbourhood: el("neighbourhood").value.trim(),
    space: Number(el("space").value),
    rooms: Number(el("rooms").value),
    bathrooms: Number(el("bathrooms").value),
    street_width: Number(el("street_width").value),
    property_age: Number(el("property_age").value),
    front: el("front").value,
    location: el("location").value,
  };
}

function validate(p){
  if(!p.neighbourhood) return "Neighbourhood required";
  if(p.space < 50) return "Area must be ≥ 50";
  if(p.rooms < 1) return "Rooms must be ≥ 1";
  if(p.bathrooms < 1) return "Bathrooms must be ≥ 1";
  if(p.street_width < 8) return "Street width must be ≥ 8";
  return null;
}

async function predict(p){
  const err = validate(p);
  el("errorBox").textContent = err || "";
  if(err) return;

  const res = await fetch(PREDICT_URL,{
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body:JSON.stringify(p)
  });

  const data = await res.json();
  el("priceValue").textContent = formatSAR(data.predicted_price);
  el("priceSub").textContent = `log(price) = ${data.log_prediction.toFixed(4)}`;

  el("miniNeighbourhood").textContent = p.neighbourhood;
  el("miniSpace").textContent = `${p.space} m²`;
  el("miniFront").textContent = p.front;
  el("miniLocation").textContent = p.location;
  el("miniPrice").textContent = formatSAR(data.predicted_price);

  addCard(data.predicted_price, p);
}

function addCard(price, p){
  const card = document.createElement("div");
  card.className = "card";

  card.innerHTML = `
    <div class="card-img" style="background-image:url('${pickHouseImage()}')"></div>
    <div class="card-body">
      <div class="card-price">${formatSAR(price)}</div>
      <div class="card-meta">${p.neighbourhood} • ${p.location} • ${p.space} m²</div>
      <div class="card-tags">
        <span class="tag">${p.rooms} rooms</span>
        <span class="tag">${p.bathrooms} baths</span>
        <span class="tag">${p.front}</span>
        <span class="tag">${p.street_width}m street</span>
      </div>
    </div>
  `;
  el("cardsGrid").prepend(card);
}

document.addEventListener("DOMContentLoaded", ()=>{
  el("predictForm").addEventListener("submit", e=>{
    e.preventDefault();
    predict(readInputs());
  });

  el("resetBtn").onclick = ()=>location.reload();
  el("clearCardsBtn").onclick = ()=>el("cardsGrid").innerHTML="";
});

