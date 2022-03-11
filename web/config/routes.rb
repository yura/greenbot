Rails.application.routes.draw do
  resources :cities
  resources :recyclers
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Defines the root path route ("/")
  root "recyclers#index"
end
