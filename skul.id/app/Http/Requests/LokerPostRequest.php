<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class LokerPostRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, \Illuminate\Contracts\Validation\ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        return [
            'nama_perusahaan' => 'required|string|max:255',
            'deskripsi' => 'required|string',
            'posisi' => 'required|string|max:255',
            'lokasi' => 'required|string|max:255',
            'tipe' => 'required|in:Part Time,Full Time, Remote',
            'pendidikan' => 'required|string|max:255',
            'gaji_min' => 'required|numeric',
            'gaji_max' => 'required|numeric|gte:gaji_min',
            'gambar' => 'required|image|mimes:jpg,jpeg,png|max:10048',
            'skills' => 'required|string|max:255',
        ];
    }
}
